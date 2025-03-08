$(document).ready(function () {
  // Trigger Attendance with AJAX
  $("#trigger-attendance").on("click", function () {
    var period_id = $("#period_id").val();
    if (!period_id) {
      $("#message").text("Please select a period.").addClass("error");
      return;
    }

    $("#message").text("Starting attendance marking...").removeClass("error");
    $.ajax({
      url: "/mark_attendance/" + period_id,
      type: "POST",
      dataType: "json",
      success: function (data) {
        if (data.status === "success") {
          $("#message")
            .text(
              data.message +
                " " +
                data.absent_message +
                " Recognized: " +
                data.recognized_count
            )
            .removeClass("error");
          $("#view-attendance").click();
        } else {
          $("#message").text(data.message).addClass("error");
        }
      },
      error: function (xhr, status, error) {
        $("#message")
          .text("Error triggering attendance: " + error)
          .addClass("error");
        console.log("Error:", error);
      },
    });
  });

  // View Attendance with AJAX
  $("#view-attendance").on("click", function () {
    var period_id = $("#period_id").val();
    if (!period_id) {
      $("#message").text("Please select a period.").addClass("error");
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
            "<tr><th>First Name</th><th>Middle Name</th><th>Last Name</th><th>Roll No</th><th>Status</th><th>Timestamp</th></tr>";
          $("#attendance-table").append(headerRow);
          $.each(data.records, function (index, record) {
            var row =
              "<tr>" +
              "<td>" +
              record.first_name +
              "</td>" +
              "<td>" +
              record.middle_name +
              "</td>" +
              "<td>" +
              record.last_name +
              "</td>" +
              "<td>" +
              record.roll_no +
              "</td>" +
              "<td>" +
              record.status +
              "</td>" +
              "<td>" +
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
        console.log("Error:", error);
      },
    });
  });
});
