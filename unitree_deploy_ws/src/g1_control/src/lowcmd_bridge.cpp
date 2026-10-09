// Copyright 2026 cjh
// Licensed under the Apache License, Version 2.0.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include "g1_policy_msgs/msg/control_status.hpp"
#include "g1_policy_msgs/msg/policy_joint_target.hpp"
#include "g1_policy_msgs/msg/proprio_policy_input.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "unitree_api/msg/request.hpp"
#include "unitree_api/msg/response.hpp"
#include "unitree_hg/msg/low_cmd.hpp"
#include "unitree_hg/msg/low_state.hpp"

// ROS 2 Foxy incorrectly assumes every message member named `header` contains
// a timestamp. Unitree API responses use a request identity header instead.
namespace libstatistics_collector::topic_statistics_collector {
template <>
struct TimeStamp<unitree_api::msg::Response> {
  static std::pair<bool, std::int64_t> value(
      const unitree_api::msg::Response & /*message*/) {
    return std::make_pair(true, 0);
  }
};
}  // namespace libstatistics_collector::topic_statistics_collector

namespace {

constexpr std::size_t kMotorCount = 35;
constexpr std::size_t kJointCount = 23;
constexpr std::uint8_t kMotorEnabled = 1;
constexpr std::uint8_t kMotorDisabled = 0;
constexpr std::uint8_t kPrMode = 0;
constexpr std::int64_t kMotionCheckModeApi = 1001;
constexpr std::int64_t kMotionReleaseModeApi = 1003;

// Unitree hardware order of the 23 motors physically present on this G1.
constexpr std::array<std::size_t, kJointCount> kActiveMotorIndices{
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
    12, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26};

// Frozen Isaac Lab policy index -> Unitree LowCmd motor index.
constexpr std::array<std::size_t, kJointCount> kPolicyToMotor{
    0, 6, 12, 1, 7, 15, 22, 2, 8, 16, 23, 3,
    9, 17, 24, 4, 10, 18, 25, 5, 11, 19, 26};

constexpr std::array<float, kJointCount> kDefaultQCompact{
    -0.2F, 0.0F,  0.0F, 0.42F, -0.23F, 0.0F,
    -0.2F, 0.0F,  0.0F, 0.42F, -0.23F, 0.0F,
    0.0F,
    0.35F, 0.18F, 0.0F, 0.87F, 0.0F,
    0.35F, -0.18F, 0.0F, 0.87F, 0.0F};

constexpr std::array<float, kJointCount> kKpCompact{
    200.0F, 150.0F, 150.0F, 200.0F, 20.0F, 20.0F,
    200.0F, 150.0F, 150.0F, 200.0F, 20.0F, 20.0F,
    200.0F,
    100.0F, 100.0F, 50.0F, 50.0F, 15.0F,
    100.0F, 100.0F, 50.0F, 50.0F, 15.0F};

constexpr std::array<float, kJointCount> kKdCompact{
    5.0F, 5.0F, 5.0F, 5.0F, 2.0F, 2.0F,
    5.0F, 5.0F, 5.0F, 5.0F, 2.0F, 2.0F,
    5.0F,
    2.0F, 2.0F, 2.0F, 2.0F, 2.0F,
    2.0F, 2.0F, 2.0F, 2.0F, 2.0F};

struct RawMotorCmd {
  std::uint8_t mode;
  float q;
  float dq;
  float tau;
  float kp;
  float kd;
  std::uint32_t reserve{0};
};

struct RawLowCmd {
  std::uint8_t mode_pr;
  std::uint8_t mode_machine;
  std::array<RawMotorCmd, kMotorCount> motor_cmd;
  std::array<std::uint32_t, 4> reserve;
  std::uint32_t crc;
};

std::uint32_t Crc32Core(std::uint32_t *data_words, std::uint32_t length) {
  std::uint32_t crc = 0xFFFFFFFFU;
  constexpr std::uint32_t polynomial = 0x04c11db7U;
  for (std::uint32_t i = 0; i < length; ++i) {
    std::uint32_t xbit = 1U << 31U;
    const std::uint32_t data = data_words[i];
    for (std::uint32_t bit = 0; bit < 32U; ++bit) {
      if ((crc & 0x80000000U) != 0U) {
        crc = (crc << 1U) ^ polynomial;
      } else {
        crc <<= 1U;
      }
      if ((data & xbit) != 0U) {
        crc ^= polynomial;
      }
      xbit >>= 1U;
    }
  }
  return crc;
}

void SetCrc(unitree_hg::msg::LowCmd &message) {
  RawLowCmd raw{};
  raw.mode_pr = message.mode_pr;
  raw.mode_machine = message.mode_machine;
  for (std::size_t i = 0; i < kMotorCount; ++i) {
    raw.motor_cmd[i].mode = message.motor_cmd[i].mode;
    raw.motor_cmd[i].q = message.motor_cmd[i].q;
    raw.motor_cmd[i].dq = message.motor_cmd[i].dq;
    raw.motor_cmd[i].tau = message.motor_cmd[i].tau;
    raw.motor_cmd[i].kp = message.motor_cmd[i].kp;
    raw.motor_cmd[i].kd = message.motor_cmd[i].kd;
    raw.motor_cmd[i].reserve = message.motor_cmd[i].reserve;
  }
  // Preserve the exact byte contract used by Unitree's motor_crc_hg.cpp.
  std::memcpy(&raw.reserve[0], &message.reserve[0], 4);
  raw.crc = Crc32Core(reinterpret_cast<std::uint32_t *>(&raw),
                      (sizeof(RawLowCmd) >> 2U) - 1U);
  message.crc = raw.crc;
}

std::uint16_t DecodeKeys(const std::array<std::uint8_t, 40> &payload) {
  return static_cast<std::uint16_t>(payload[2]) |
         (static_cast<std::uint16_t>(payload[3]) << 8U);
}

bool Rising(std::uint16_t previous, std::uint16_t current, int bit) {
  const auto mask = static_cast<std::uint16_t>(1U << bit);
  return (current & mask) != 0U && (previous & mask) == 0U;
}

bool ExtractJsonString(const std::string &json, const std::string &field,
                       std::string *value) {
  const std::string key = "\"" + field + "\"";
  const auto key_position = json.find(key);
  if (key_position == std::string::npos) {
    return false;
  }
  const auto colon_position = json.find(':', key_position + key.size());
  if (colon_position == std::string::npos) {
    return false;
  }
  const auto quote_position = json.find('"', colon_position + 1U);
  if (quote_position == std::string::npos) {
    return false;
  }
  std::string parsed;
  bool escaped = false;
  for (std::size_t i = quote_position + 1U; i < json.size(); ++i) {
    const char character = json[i];
    if (escaped) {
      parsed.push_back(character);
      escaped = false;
    } else if (character == '\\') {
      escaped = true;
    } else if (character == '"') {
      *value = parsed;
      return true;
    } else {
      parsed.push_back(character);
    }
  }
  return false;
}

enum class BridgeState : std::uint8_t {
  kWaiting = 0,
  kAcquiring = 1,
  kStanding = 2,
  kHolding = 3,
  kPolicy = 4,
  kDamping = 5,
};

const char *StateName(BridgeState state) {
  switch (state) {
    case BridgeState::kWaiting:
      return "WAITING";
    case BridgeState::kAcquiring:
      return "ACQUIRING";
    case BridgeState::kStanding:
      return "STANDING";
    case BridgeState::kHolding:
      return "HOLDING";
    case BridgeState::kPolicy:
      return "POLICY";
    case BridgeState::kDamping:
      return "DAMPING";
  }
  return "UNKNOWN";
}

}  // namespace

class G1LowCmdBridge : public rclcpp::Node {
 public:
  G1LowCmdBridge() : Node("g1_lowcmd_bridge") {
    const auto lowstate_topic =
        declare_parameter<std::string>("lowstate_topic", "/lowstate");
    const auto lowcmd_topic =
        declare_parameter<std::string>("lowcmd_topic", "/lowcmd");
    const auto target_topic = declare_parameter<std::string>(
        "target_topic", "/g1/policy/joint_target");
    const auto event_topic =
        declare_parameter<std::string>("event_topic", "/g1/control/event");
    const auto status_topic = declare_parameter<std::string>(
        "status_topic", "/g1/control/status");
    const auto proprio_topic = declare_parameter<std::string>(
        "proprio_topic", "/g1/proprio/policy_input");
    const double publish_rate = declare_parameter<double>("publish_rate", 500.0);
    const double status_rate = declare_parameter<double>("status_rate", 50.0);
    const double proprio_rate = declare_parameter<double>("proprio_rate", 50.0);
    dry_run_ = declare_parameter<bool>("dry_run", true);
    auto_release_motion_service_ =
        declare_parameter<bool>("auto_release_motion_service", false);
    motion_request_topic_ = declare_parameter<std::string>(
        "motion_request_topic", "/api/motion_switcher/request");
    motion_response_topic_ = declare_parameter<std::string>(
        "motion_response_topic", "/api/motion_switcher/response");
    motion_response_timeout_ =
        declare_parameter<double>("motion_response_timeout", 5.0);
    motion_retry_period_ =
        declare_parameter<double>("motion_retry_period", 2.0);
    remote_buttons_enabled_ =
        declare_parameter<bool>("remote_buttons_enabled", false);
    start_button_bit_ = declare_parameter<int>("start_button_bit", 2);
    select_button_bit_ = declare_parameter<int>("select_button_bit", 3);
    policy_button_bit_ = declare_parameter<int>("policy_button_bit", 8);
    stand_duration_ = declare_parameter<double>("stand_duration", 3.0);
    damping_kd_ = declare_parameter<double>("damping_kd", 10.0);
    if (publish_rate <= 0.0 || status_rate <= 0.0 || proprio_rate <= 0.0 ||
        stand_duration_ <= 0.0 || motion_response_timeout_ <= 0.0 ||
        motion_retry_period_ <= 0.0) {
      throw std::invalid_argument(
          "rates, durations, and timeouts must be positive");
    }
    if (!dry_run_ && !auto_release_motion_service_) {
      throw std::invalid_argument(
          "real LowCmd output requires auto_release_motion_service=true");
    }
    for (int bit : {start_button_bit_, select_button_bit_, policy_button_bit_}) {
      if (bit < 0 || bit > 15) {
        throw std::invalid_argument("remote button bits must be in [0, 15]");
      }
    }

    default_q_.fill(0.0F);
    kp_.fill(0.0F);
    kd_.fill(0.0F);
    for (std::size_t i = 0; i < kJointCount; ++i) {
      const auto motor = kActiveMotorIndices[i];
      default_q_[motor] = kDefaultQCompact[i];
      kp_[motor] = kKpCompact[i];
      kd_[motor] = kKdCompact[i];
    }

    lowstate_subscription_ =
        create_subscription<unitree_hg::msg::LowState>(
            lowstate_topic, 10,
            std::bind(&G1LowCmdBridge::OnLowState, this,
                      std::placeholders::_1));
    target_subscription_ =
        create_subscription<g1_policy_msgs::msg::PolicyJointTarget>(
            target_topic, 10,
            std::bind(&G1LowCmdBridge::OnTarget, this,
                      std::placeholders::_1));
    lowcmd_publisher_ =
        create_publisher<unitree_hg::msg::LowCmd>(lowcmd_topic, 10);
    event_publisher_ =
        create_publisher<std_msgs::msg::String>(event_topic, 10);
    status_publisher_ =
        create_publisher<g1_policy_msgs::msg::ControlStatus>(status_topic, 10);
    proprio_publisher_ =
        create_publisher<g1_policy_msgs::msg::ProprioPolicyInput>(
            proprio_topic, 1);
    const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(1.0 / publish_rate));
    timer_ = create_wall_timer(period, std::bind(&G1LowCmdBridge::Control, this));
    const auto status_period =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(1.0 / status_rate));
    status_timer_ = create_wall_timer(
        status_period, std::bind(&G1LowCmdBridge::PublishStatus, this));
    const auto proprio_period =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(1.0 / proprio_rate));
    proprio_timer_ = create_wall_timer(
        proprio_period, std::bind(&G1LowCmdBridge::PublishProprio, this));

    if (!dry_run_) {
      motion_request_publisher_ =
          create_publisher<unitree_api::msg::Request>(motion_request_topic_, 1);
      motion_response_subscription_ =
          create_subscription<unitree_api::msg::Response>(
              motion_response_topic_, 1,
              std::bind(&G1LowCmdBridge::OnMotionResponse, this,
                        std::placeholders::_1));
      motion_thread_ = std::thread(&G1LowCmdBridge::MotionReleaseWorker, this);
    }

    RCLCPP_INFO(
        get_logger(),
        "500 Hz LowCmd bridge ready: lowstate=%s proprio=%s@%.1fHz "
        "target=%s lowcmd=%s dry_run=%s remote_buttons=%s "
        "auto_release=%s state=%s",
        lowstate_topic.c_str(), proprio_topic.c_str(), proprio_rate,
        target_topic.c_str(), lowcmd_topic.c_str(),
        dry_run_ ? "true" : "false",
        remote_buttons_enabled_ ? "true" : "false",
        auto_release_motion_service_ ? "true" : "false", StateName(state_));
  }

  ~G1LowCmdBridge() override {
    stop_motion_thread_.store(true);
    arm_condition_.notify_all();
    response_condition_.notify_all();
    if (motion_thread_.joinable()) {
      motion_thread_.join();
    }
  }

 private:
  void BeginStandTransition() {
    initial_q_ = measured_q_;
    stand_start_ = std::chrono::steady_clock::now();
    state_ = BridgeState::kStanding;
    RCLCPP_INFO(get_logger(),
                "low-level ownership ready: entering %.2f s stand transition",
                stand_duration_);
  }

  void PublishEvent(const std::string &event) {
    std_msgs::msg::String message;
    message.data = event;
    event_publisher_->publish(message);
  }

  void OnTarget(
      const g1_policy_msgs::msg::PolicyJointTarget::SharedPtr message) {
    const rclcpp::Time candidate_stamp(message->header.stamp);
    if (policy_request_pending_ && state_ == BridgeState::kHolding &&
        candidate_stamp < policy_request_stamp_) {
      RCLCPP_WARN(get_logger(),
                  "ignoring policy target older than the A-button request");
      return;
    }
    policy_target_ = message->q_target;
    target_sequence_ = message->sequence;
    target_stamp_ = candidate_stamp;
    have_policy_target_ = true;
    if (policy_request_pending_ && state_ == BridgeState::kHolding) {
      policy_request_pending_ = false;
      state_ = BridgeState::kPolicy;
      PublishEvent("policy_running");
      RCLCPP_INFO(
          get_logger(),
          "fresh post-A policy target accepted (sequence=%lu); tracking enabled",
          static_cast<unsigned long>(target_sequence_));
    }
  }

  void OnMotionResponse(
      const unitree_api::msg::Response::SharedPtr message) {
    std::lock_guard<std::mutex> lock(response_mutex_);
    if (!response_waiting_ ||
        message->header.identity.id != pending_request_id_) {
      return;
    }
    response_code_ = message->header.status.code;
    response_data_ = message->data;
    response_waiting_ = false;
    response_condition_.notify_all();
  }

  int CallMotionApi(std::int64_t api_id, std::string *response_data) {
    unitree_api::msg::Request request;
    request.header.identity.api_id = api_id;
    request.header.identity.id =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count();
    {
      std::lock_guard<std::mutex> lock(response_mutex_);
      pending_request_id_ = request.header.identity.id;
      response_code_ = -1;
      response_data_.clear();
      response_waiting_ = true;
    }
    motion_request_publisher_->publish(request);

    std::unique_lock<std::mutex> lock(response_mutex_);
    const bool received = response_condition_.wait_for(
        lock, std::chrono::duration<double>(motion_response_timeout_), [this] {
          return !response_waiting_ || stop_motion_thread_.load();
        });
    if (!received || stop_motion_thread_.load()) {
      response_waiting_ = false;
      return -1;
    }
    if (response_data != nullptr) {
      *response_data = response_data_;
    }
    return response_code_;
  }

  bool WaitMotionRetry() {
    std::unique_lock<std::mutex> lock(arm_mutex_);
    return arm_condition_.wait_for(
        lock, std::chrono::duration<double>(motion_retry_period_),
        [this] { return stop_motion_thread_.load(); });
  }

  void MotionReleaseWorker() {
    {
      std::unique_lock<std::mutex> lock(arm_mutex_);
      arm_condition_.wait(lock, [this] {
        return arm_requested_.load() || stop_motion_thread_.load();
      });
    }
    if (stop_motion_thread_.load()) {
      return;
    }

    RCLCPP_WARN(get_logger(),
                "START accepted: checking and releasing the Unitree motion service before LowCmd takeover");
    while (rclcpp::ok() && !stop_motion_thread_.load()) {
      std::string data;
      const int check_result = CallMotionApi(kMotionCheckModeApi, &data);
      if (check_result != 0) {
        RCLCPP_ERROR(get_logger(),
                     "MotionSwitcher CheckMode failed/timeout (code=%d); LowCmd remains inhibited",
                     check_result);
        if (WaitMotionRetry()) {
          return;
        }
        continue;
      }

      std::string mode_name;
      if (!ExtractJsonString(data, "name", &mode_name)) {
        RCLCPP_ERROR(get_logger(),
                     "MotionSwitcher CheckMode returned an invalid payload; LowCmd remains inhibited");
        if (WaitMotionRetry()) {
          return;
        }
        continue;
      }
      if (mode_name.empty()) {
        motion_service_ready_.store(true);
        RCLCPP_INFO(get_logger(),
                    "MotionSwitcher confirms release mode; 500 Hz LowCmd takeover is enabled");
        return;
      }

      RCLCPP_WARN(get_logger(),
                  "Unitree motion service '%s' is active; requesting ReleaseMode",
                  mode_name.c_str());
      const int release_result = CallMotionApi(kMotionReleaseModeApi, nullptr);
      if (release_result != 0) {
        RCLCPP_ERROR(get_logger(),
                     "MotionSwitcher ReleaseMode failed/timeout (code=%d); LowCmd remains inhibited",
                     release_result);
      } else {
        RCLCPP_INFO(get_logger(),
                    "ReleaseMode accepted; verifying that no motion service remains active");
      }
      if (WaitMotionRetry()) {
        return;
      }
    }
  }

  void OnLowState(const unitree_hg::msg::LowState::SharedPtr message) {
    have_lowstate_ = true;
    last_lowstate_stamp_ = get_clock()->now();
    mode_machine_ = message->mode_machine;
    for (std::size_t i = 0; i < kMotorCount; ++i) {
      measured_q_[i] = message->motor_state[i].q;
      measured_dq_[i] = message->motor_state[i].dq;
    }
    angular_velocity_ = message->imu_state.gyroscope;
    quaternion_wxyz_ = message->imu_state.quaternion;
    wireless_remote_ = message->wireless_remote;
    const std::uint16_t keys = DecodeKeys(message->wireless_remote);
    if (remote_buttons_enabled_) {
      if (Rising(previous_keys_, keys, select_button_bit_) &&
          state_ != BridgeState::kWaiting) {
        policy_request_pending_ = false;
        have_policy_target_ = false;
        if (state_ == BridgeState::kAcquiring &&
            !motion_service_ready_.load()) {
          damping_after_acquire_ = true;
        } else {
          state_ = BridgeState::kDamping;
        }
        PublishEvent("policy_exit");
        RCLCPP_WARN(get_logger(),
                    "SELECT pressed: controlled damping is latched%s; restart is required to re-arm",
                    damping_after_acquire_ ? " for takeover completion" : "");
      } else if (Rising(previous_keys_, keys, start_button_bit_) &&
                 state_ == BridgeState::kWaiting) {
        if (dry_run_) {
          BeginStandTransition();
        } else {
          state_ = BridgeState::kAcquiring;
          arm_requested_.store(true);
          arm_condition_.notify_all();
          RCLCPP_INFO(get_logger(),
                      "START pressed: waiting for verified MotionSwitcher release before sending LowCmd");
        }
      } else if (Rising(previous_keys_, keys, policy_button_bit_) &&
                 state_ == BridgeState::kHolding) {
        // Discard every target produced before A. The policy process uses the
        // request event to clear its observation/action history and publishes
        // sequence zero from the next 50 Hz proprioception snapshot.
        have_policy_target_ = false;
        target_sequence_ = 0U;
        target_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
        policy_request_stamp_ = get_clock()->now();
        policy_request_pending_ = true;
        PublishEvent("policy_requested");
        RCLCPP_INFO(get_logger(),
                    "A pressed: holding default posture while waiting for a fresh policy epoch");
      }
    }
    previous_keys_ = keys;
  }

  void FillDamping(unitree_hg::msg::LowCmd &command) {
    for (const auto motor : kActiveMotorIndices) {
      auto &item = command.motor_cmd[motor];
      item.mode = kMotorEnabled;
      item.q = 0.0F;
      item.dq = 0.0F;
      item.tau = 0.0F;
      item.kp = 0.0F;
      item.kd = static_cast<float>(damping_kd_);
    }
  }

  std::array<float, kMotorCount> DesiredPosition() {
    std::array<float, kMotorCount> desired = default_q_;
    if (state_ == BridgeState::kStanding) {
      const double elapsed = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - stand_start_)
                                 .count();
      const float alpha = static_cast<float>(
          std::clamp(elapsed / stand_duration_, 0.0, 1.0));
      for (const auto motor : kActiveMotorIndices) {
        desired[motor] =
            (1.0F - alpha) * initial_q_[motor] + alpha * default_q_[motor];
      }
      if (alpha >= 1.0F) {
        state_ = BridgeState::kHolding;
        RCLCPP_INFO(get_logger(),
                    "stand transition complete; holding training default posture");
      }
    } else if (state_ == BridgeState::kPolicy) {
      for (std::size_t policy_index = 0; policy_index < kJointCount;
           ++policy_index) {
        desired[kPolicyToMotor[policy_index]] = policy_target_[policy_index];
      }
    }
    return desired;
  }

  void FillPositionCommand(unitree_hg::msg::LowCmd &command) {
    const auto desired = DesiredPosition();
    for (std::size_t compact_index = 0; compact_index < kJointCount;
         ++compact_index) {
      const auto motor = kActiveMotorIndices[compact_index];
      auto &item = command.motor_cmd[motor];
      item.mode = kMotorEnabled;
      item.q = desired[motor];
      item.dq = 0.0F;
      item.tau = 0.0F;
      item.kp = kp_[motor];
      item.kd = kd_[motor];
    }
  }

  void Control() {
    if (!have_lowstate_ || state_ == BridgeState::kWaiting) {
      return;
    }
    if (state_ == BridgeState::kAcquiring) {
      if (!motion_service_ready_.load()) {
        return;
      }
      if (damping_after_acquire_) {
        state_ = BridgeState::kDamping;
        RCLCPP_WARN(get_logger(),
                    "motion service released after SELECT; starting directly in controlled damping");
      } else {
        BeginStandTransition();
      }
    }
    unitree_hg::msg::LowCmd command;
    command.mode_pr = kPrMode;
    command.mode_machine = mode_machine_;
    for (auto &item : command.motor_cmd) {
      item.mode = kMotorDisabled;
      item.q = 0.0F;
      item.dq = 0.0F;
      item.tau = 0.0F;
      item.kp = 0.0F;
      item.kd = 0.0F;
      item.reserve = 0U;
    }
    if (state_ == BridgeState::kDamping) {
      FillDamping(command);
    } else {
      FillPositionCommand(command);
    }
    if (!dry_run_) {
      SetCrc(command);
      lowcmd_publisher_->publish(command);
    }
  }

  void PublishStatus() {
    g1_policy_msgs::msg::ControlStatus message;
    message.header.stamp = get_clock()->now();
    message.header.frame_id = "g1_lowcmd_bridge";
    message.state = static_cast<std::uint8_t>(state_);
    message.dry_run = dry_run_;
    message.motion_service_released = motion_service_ready_.load();
    message.mode_pr = kPrMode;
    message.mode_machine = mode_machine_;
    message.target_sequence = target_sequence_;
    message.target_age = -1.0F;
    if (have_policy_target_) {
      message.target_age = static_cast<float>(
          (get_clock()->now() - target_stamp_).seconds());
    }
    status_publisher_->publish(message);
  }

  void PublishProprio() {
    if (!have_lowstate_) {
      return;
    }
    g1_policy_msgs::msg::ProprioPolicyInput message;
    message.header.stamp = last_lowstate_stamp_;
    message.header.frame_id = "torso_link";
    message.sequence = proprio_sequence_++;
    message.angular_velocity = angular_velocity_;
    message.quaternion_wxyz = quaternion_wxyz_;
    message.wireless_remote = wireless_remote_;
    for (std::size_t policy_index = 0; policy_index < kJointCount;
         ++policy_index) {
      const auto motor = kPolicyToMotor[policy_index];
      message.joint_position[policy_index] = measured_q_[motor];
      message.joint_velocity[policy_index] = measured_dq_[motor];
    }
    proprio_publisher_->publish(message);
  }

  rclcpp::Subscription<unitree_hg::msg::LowState>::SharedPtr
      lowstate_subscription_;
  rclcpp::Subscription<g1_policy_msgs::msg::PolicyJointTarget>::SharedPtr
      target_subscription_;
  rclcpp::Publisher<unitree_hg::msg::LowCmd>::SharedPtr lowcmd_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr event_publisher_;
  rclcpp::Publisher<g1_policy_msgs::msg::ControlStatus>::SharedPtr
      status_publisher_;
  rclcpp::Publisher<g1_policy_msgs::msg::ProprioPolicyInput>::SharedPtr
      proprio_publisher_;
  rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr
      motion_request_publisher_;
  rclcpp::Subscription<unitree_api::msg::Response>::SharedPtr
      motion_response_subscription_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::TimerBase::SharedPtr status_timer_;
  rclcpp::TimerBase::SharedPtr proprio_timer_;

  bool dry_run_{true};
  bool auto_release_motion_service_{false};
  bool remote_buttons_enabled_{false};
  bool have_lowstate_{false};
  bool have_policy_target_{false};
  bool policy_request_pending_{false};
  bool damping_after_acquire_{false};
  int start_button_bit_{2};
  int select_button_bit_{3};
  int policy_button_bit_{8};
  double stand_duration_{3.0};
  double damping_kd_{10.0};
  double motion_response_timeout_{5.0};
  double motion_retry_period_{2.0};
  std::string motion_request_topic_;
  std::string motion_response_topic_;
  std::uint8_t mode_machine_{0};
  std::uint16_t previous_keys_{0};
  BridgeState state_{BridgeState::kWaiting};
  std::chrono::steady_clock::time_point stand_start_{};
  std::array<float, kMotorCount> measured_q_{};
  std::array<float, kMotorCount> measured_dq_{};
  std::array<float, kMotorCount> initial_q_{};
  std::array<float, kMotorCount> default_q_{};
  std::array<float, kMotorCount> kp_{};
  std::array<float, kMotorCount> kd_{};
  std::array<float, 3> angular_velocity_{};
  std::array<float, 4> quaternion_wxyz_{};
  std::array<std::uint8_t, 40> wireless_remote_{};
  std::array<float, kJointCount> policy_target_{};
  std::uint64_t target_sequence_{0};
  rclcpp::Time target_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time policy_request_stamp_{0, 0, RCL_ROS_TIME};
  std::uint64_t proprio_sequence_{0};
  rclcpp::Time last_lowstate_stamp_{0, 0, RCL_ROS_TIME};

  std::atomic<bool> arm_requested_{false};
  std::atomic<bool> motion_service_ready_{false};
  std::atomic<bool> stop_motion_thread_{false};
  std::thread motion_thread_;
  std::mutex arm_mutex_;
  std::condition_variable arm_condition_;
  std::mutex response_mutex_;
  std::condition_variable response_condition_;
  bool response_waiting_{false};
  std::int64_t pending_request_id_{0};
  int response_code_{-1};
  std::string response_data_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<G1LowCmdBridge>());
  rclcpp::shutdown();
  return 0;
}
