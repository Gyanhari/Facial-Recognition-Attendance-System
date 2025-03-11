// Filter periods by date
function filterPeriods() {
  const dateFilter = document.getElementById("date_filter").value;
  const periodSelect = document.getElementById("period_id");
  const options = periodSelect.getElementsByTagName("option");

  for (let i = 0; i < options.length; i++) {
      const option = options[i];
      const optionDate = option.getAttribute("data-date");
      option.style.display = (!dateFilter || optionDate === dateFilter) ? "block" : "none";
  }
  if (periodSelect.value && periodSelect.options[periodSelect.selectedIndex].style.display === "none") {
      periodSelect.value = "";
  }
  updateAttendanceButtonState();
}

// Filter attendance records by status and date
function filterAttendance() {
  const statusFilter = document.getElementById("status_filter").value.toLowerCase();
  const dateFilter = document.getElementById("date_filter").value;
  const rows = document.querySelectorAll("#attendance-table tbody tr");

  rows.forEach(row => {
      const status = row.cells[4].textContent.toLowerCase();
      const timestamp = row.cells[5].textContent;
      const rowDate = timestamp.split(" ")[0]; // Extract date from timestamp (e.g., "2025-03-10")

      const matchesStatus = !statusFilter || status === statusFilter;
      const matchesDate = !dateFilter || rowDate === dateFilter;

      row.style.display = (matchesStatus && matchesDate) ? "" : "none";
  });

  const visibleRows = Array.from(rows).filter(row => row.style.display !== "none");
  document.getElementById("attendance-table").style.display = visibleRows.length > 0 ? "" : "none";
  if (visibleRows.length === 0 && rows.length > 0) {
      document.getElementById("message").innerHTML = '<div class="alert alert-info">No records match the selected filters.</div>';
  }
}

// Check period status and unmarked students, then update button state
function updateAttendanceButtonState() {
  const periodId = $('#period_id').val();
  const triggerButton = $('#trigger-attendance');
  const messageDiv = $('#message');

  if (!periodId) {
      triggerButton.prop('disabled', true);
      messageDiv.html('<div class="alert alert-warning">Please select a period.</div>');
      return;
  }

  $.ajax({
      url: `/check_period_status/${periodId}`,
      type: 'GET',
      success: function(response) {
          if (response.status === 'success') {
              const { can_mark, message } = response;
              triggerButton.prop('disabled', !can_mark);
              messageDiv.html(`<div class="alert alert-${can_mark ? 'info' : 'warning'}">${message}</div>`);
          } else {
              triggerButton.prop('disabled', true);
              messageDiv.html(`<div class="alert alert-danger">${response.messages[0].text}</div>`);
          }
      },
      error: function(xhr) {
          triggerButton.prop('disabled', true);
          messageDiv.html(`<div class="alert alert-danger">${xhr.responseJSON?.message || 'Failed to check period status'}</div>`);
      }
  });
}

// Trigger attendance marking
$('#trigger-attendance').on('click', function() {
  const periodId = $('#period_id').val();
  const messageDiv = $('#message');

  if (!periodId) {
      messageDiv.html('<div class="alert alert-danger">Please select a period.</div>');
      return;
  }

  messageDiv.html('<div class="alert alert-info">Processing attendance... (This may take up to 5 minutes)</div>');

  $.ajax({
      url: `/mark_attendance/${periodId}`,
      type: 'POST',
      timeout: 310000, // 5 minutes + 10 seconds
      success: function(response) {
          messageDiv.empty();
          if (response.status === 'success' && response.messages) {
              response.messages.forEach(msg => {
                  const alertClass = `alert-${msg.category === 'success' ? 'success' : msg.category === 'info' ? 'info' : 'danger'}`;
                  messageDiv.append(`<div class="alert ${alertClass}">${msg.text}</div>`);
              });
              viewAttendance(periodId); // Refresh table to show current state
              updateAttendanceButtonState(); // Update button state
          } else {
              messageDiv.append(`<div class="alert alert-danger">${response.messages[0]?.text || 'An error occurred'}</div>`);
          }
      },
      error: function(xhr) {
          messageDiv.empty();
          const response = xhr.responseJSON || { messages: [{ text: 'Unknown error' }] };
          messageDiv.append(`<div class="alert alert-danger">${response.messages[0].text || 'Failed to process attendance'}</div>`);
      }
  });
});

// View attendance records
$('#view-attendance').on('click', function() {
  const periodId = $('#period_id').val();
  if (!periodId) {
      $('#message').html('<div class="alert alert-danger">Please select a period.</div>');
      return;
  }
  viewAttendance(periodId);
});

// Fetch and display attendance records
function viewAttendance(periodId) {
  $.ajax({
      url: `/get_attendance/${periodId}`,
      type: 'GET',
      success: function(data) {
          $('#message').html('<div class="alert alert-info">Attendance records loaded.</div>');
          const tbody = $('#attendance-table tbody');
          tbody.empty();
          $('#attendance-table').show();

          data.records.forEach(record => {
              const row = `
                  <tr>
                      <td>${record.first_name}</td>
                      <td>${record.middle_name || ''}</td>
                      <td>${record.last_name}</td>
                      <td>${record.roll_no}</td>
                      <td>${record.status}</td>
                      <td>${record.timestamp}</td>
                  </tr>`;
              tbody.append(row);
          });
          filterAttendance();
      },
      error: function(xhr) {
          $('#message').html(`<div class="alert alert-danger">${xhr.responseJSON?.message || 'Failed to fetch attendance'}</div>`);
          $('#attendance-table').hide();
      }
  });
}

// Initial setup
$(document).ready(function() {
  filterPeriods(); // Filter periods and set initial button state
  $('#period_id').on('change', function() {
      updateAttendanceButtonState(); // Update button state on period change
  });
});