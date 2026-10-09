// Copyright 2026 cjh
// Licensed under the Apache License, Version 2.0.

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "unitree_api/msg/request.hpp"
#include "unitree_api/msg/response.hpp"

// Unitree ResponseHeader has an identity/status pair rather than a ROS stamp.
// Foxy otherwise instantiates its generic stamped-message statistics helper
// and fails at compile time. This is the same specialization used by the
// official unitree_ros2 audio client example.
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

constexpr std::int64_t kAudioTtsApiId = 1001;

const std::map<std::string, std::string> kEventPhrases{
    {"policy_running", "策略运行"},
    {"visual_mode", "视觉模式"},
    {"lidar_mode", "雷达模式"},
    {"policy_exit", "策略退出"},
};

struct PendingRequest {
  std::string event;
  std::chrono::steady_clock::time_point sent_at;
};

}  // namespace

class G1VoiceAnnouncer : public rclcpp::Node {
 public:
  G1VoiceAnnouncer() : Node("g1_voice_announcer") {
    const auto event_topic =
        declare_parameter<std::string>("event_topic", "/g1/control/event");
    const auto request_topic = declare_parameter<std::string>(
        "voice_request_topic", "/api/voice/request");
    const auto response_topic = declare_parameter<std::string>(
        "voice_response_topic", "/api/voice/response");
    speaker_id_ = declare_parameter<int>("speaker_id", 0);
    response_timeout_ =
        declare_parameter<double>("response_timeout", 5.0);
    if (response_timeout_ <= 0.0) {
      throw std::invalid_argument("response_timeout must be positive");
    }

    request_publisher_ =
        create_publisher<unitree_api::msg::Request>(request_topic, 1);
    response_subscription_ =
        create_subscription<unitree_api::msg::Response>(
            response_topic, rclcpp::QoS(10).best_effort(),
            std::bind(&G1VoiceAnnouncer::OnResponse, this,
                      std::placeholders::_1));
    event_subscription_ = create_subscription<std_msgs::msg::String>(
        event_topic, 10,
        std::bind(&G1VoiceAnnouncer::OnEvent, this, std::placeholders::_1));
    timeout_timer_ = create_wall_timer(
        std::chrono::seconds(1),
        std::bind(&G1VoiceAnnouncer::ExpireRequests, this));

    RCLCPP_INFO(get_logger(),
                "G1 native TTS announcer ready: event=%s request=%s response=%s "
                "speaker_id=%d",
                event_topic.c_str(), request_topic.c_str(),
                response_topic.c_str(), speaker_id_);
  }

 private:
  static std::int64_t NewIdentity() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }

  std::string MakeParameter(const std::string &phrase) {
    std::ostringstream stream;
    stream << "{\"index\":" << tts_index_++ << ",\"text\":\""
           << phrase << "\",\"speaker_id\":" << speaker_id_ << "}";
    return stream.str();
  }

  void OnEvent(const std_msgs::msg::String::SharedPtr message) {
    // Internal handshake. "policy_running" is announced only after the first
    // freshly reset target has actually been accepted by the C++ bridge.
    if (message->data == "policy_requested") {
      return;
    }
    const auto phrase = kEventPhrases.find(message->data);
    if (phrase == kEventPhrases.end()) {
      RCLCPP_WARN(get_logger(), "ignoring unknown control event: %s",
                  message->data.c_str());
      return;
    }

    unitree_api::msg::Request request;
    request.header.identity.id = NewIdentity();
    request.header.identity.api_id = kAudioTtsApiId;
    request.header.policy.noreply = false;
    request.parameter = MakeParameter(phrase->second);
    pending_[request.header.identity.id] =
        PendingRequest{message->data, std::chrono::steady_clock::now()};
    request_publisher_->publish(request);
    RCLCPP_INFO(get_logger(), "TTS requested: event=%s phrase=%s id=%ld",
                message->data.c_str(), phrase->second.c_str(),
                static_cast<long>(request.header.identity.id));
  }

  void OnResponse(const unitree_api::msg::Response::SharedPtr response) {
    const auto pending = pending_.find(response->header.identity.id);
    if (pending == pending_.end()) {
      return;
    }
    if (response->header.status.code == 0) {
      RCLCPP_INFO(get_logger(), "TTS accepted: event=%s id=%ld",
                  pending->second.event.c_str(),
                  static_cast<long>(response->header.identity.id));
    } else {
      RCLCPP_ERROR(get_logger(), "TTS failed: event=%s id=%ld status=%d",
                   pending->second.event.c_str(),
                   static_cast<long>(response->header.identity.id),
                   response->header.status.code);
    }
    pending_.erase(pending);
  }

  void ExpireRequests() {
    const auto now = std::chrono::steady_clock::now();
    for (auto iterator = pending_.begin(); iterator != pending_.end();) {
      const double age = std::chrono::duration<double>(
                             now - iterator->second.sent_at)
                             .count();
      if (age > response_timeout_) {
        RCLCPP_WARN(get_logger(), "TTS response timeout: event=%s id=%ld",
                    iterator->second.event.c_str(),
                    static_cast<long>(iterator->first));
        iterator = pending_.erase(iterator);
      } else {
        ++iterator;
      }
    }
  }

  rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr request_publisher_;
  rclcpp::Subscription<unitree_api::msg::Response>::SharedPtr
      response_subscription_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr event_subscription_;
  rclcpp::TimerBase::SharedPtr timeout_timer_;
  std::map<std::int64_t, PendingRequest> pending_;
  std::uint32_t tts_index_{0};
  int speaker_id_{0};
  double response_timeout_{5.0};
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<G1VoiceAnnouncer>());
  rclcpp::shutdown();
  return 0;
}
