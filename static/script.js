$(document).ready(function () {
  // Trigger Attendance with AJAX
  $("#trigger-attendance").on("click", function () {
    var period_id = $("#period_id").val();
    if (!period_id) {
      $("#message").text("Please select a period.").addClass("error");
      $("#screen-detection-message").text("");
      return;
    }

    $("#message").text("Starting attendance marking...").removeClass("error");
    $("#screen-detection-message").text("");
    $.ajax({
      url: "/mark_attendance/" + period_id,
      type: "POST",
      dataType: "json",
      success: function (data) {
        if (data.status === "success") {
          $("#message")
            .text(data.message + " " + data.absent_message + " Recognized: " + data.recognized_count)
            .removeClass("error");
          if (data.screen_detected_count > 0) {
            $("#screen-detection-message").text(
              "Warning: Skipped " + data.screen_detected_count + " face(s) detected on screens."
            );
          } else {
            $("#screen-detection-message").text("No screens detected.");
          }
          $("#view-attendance").click(); // Auto-refresh attendance view
        } else {
          $("#message").text(data.message).addClass("error");
          $("#screen-detection-message").text("");
        }
      },
      error: function (xhr, status, error) {
        $("#message")
          .text("Error triggering attendance: " + error)
          .addClass("error");
        $("#screen-detection-message").text("");
        console.log("Error:", error);
      },
    });
  });

  // View Attendance with AJAX
  $("#view-attendance").on("click", function () {
    var period_id = $("#period_id").val();
    if (!period_id) {
      $("#message").text("Please select a period.").addClass("error");
      $("#screen-detection-message").text("");
      return;
    }

    $.ajax({
      url: "/get_attendance/" + period_id,
      type: "GET",
      dataType: "json",
      success: function (data) {
        $("#attendance-table").empty();
        if (data.records.length > 0) {
          $("#attendance-heading").text(
            "Attendance Record For " + data.course_name + " on " + data.date
          );
          $("#attendance-table").show();
          var headerRow =
            "<tr style='background-color: #f2f2f2;'><th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>First Name</th><th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>Middle Name</th><th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>Last Name</th><th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>Roll No</th><th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>Status</th><th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>Timestamp</th></tr>";
          $("#attendance-table").append(headerRow);
          $.each(data.records, function (index, record) {
            var row =
              "<tr>" +
              "<td style='border: 1px solid #ddd; padding: 8px;'>" +
              record.first_name +
              "</td>" +
              "<td style='border: 1px solid #ddd; padding: 8px;'>" +
              record.middle_name +
              "</td>" +
              "<td style='border: 1px solid #ddd; padding: 8px;'>" +
              record.last_name +
              "</td>" +
              "<td style='border: 1px solid #ddd; padding: 8px;'>" +
              record.roll_no +
              "</td>" +
              "<td style='border: 1px solid #ddd; padding: 8px;'>" +
              record.status +
              "</td>" +
              "<td style='border: 1px solid #ddd; padding: 8px;'>" +
              record.timestamp +
              "</td>" +
              "</tr>";
            $("#attendance-table").append(row);
          });
          $("#message").text("").removeClass("error");
        } else {
          $("#attendance-heading").text(
            "Attendance Record For " + data.course_name + " on " + data.date
          );
          $("#attendance-table").hide();
          $("#message")
            .text("No attendance records found.")
            .removeClass("error");
        }
      },
      error: function (xhr, status, error) {
        $("#message")
          .text("Error fetching attendance: " + error)
          .addClass("error");
        $("#screen-detection-message").text("");
        console.log("Error:", error);
      },
    });
  });
});