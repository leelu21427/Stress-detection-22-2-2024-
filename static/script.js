document.getElementById("predictionForm").addEventListener("submit", function(event) {
  event.preventDefault(); // Prevent the form from submitting

  // Display the prediction text
  document.getElementById("predictionResult").style.display = "block";

  // Hide the prediction text after 5 seconds
  setTimeout(function() {
    document.getElementById("predictionResult").style.display = "none";
    
    // Trigger the chatbot after displaying the prediction result
    // Load the chatbot script dynamically
    var script = document.createElement("script");
    script.src = "";
    script.async = true;
    document.body.appendChild(script);
  }, 5000); // 5 seconds = 5000 milliseconds
});


/*document.getElementById("predictionForm").addEventListener("submit", function(event) {
  event.preventDefault(); // Prevent the form from submitting

  // Trigger the chatbot after displaying the prediction result
  // Load the chatbot script dynamically
  var script = document.createElement("script");
  script.src = "//code.tidio.co/4yzyk7vfal8uvcpc9vceblbinc53efy7.js";
  script.async = true;
  document.body.appendChild(script);

  // Hide the prediction text after 15 minutes (900000 milliseconds)
  setTimeout(function() {
    console.log("Hey there! 👋 Need someone to talk to? I am Halene...");
  }, 1000); // 15 minutes = 900000 milliseconds
});*/


/*document.getElementById("predictionForm").addEventListener("submit", function(event) {
  event.preventDefault(); // Prevent the form from submitting

  // Display the prediction text
  document.getElementById("predictionResult").style.display = "block";

  // Submit the form
  this.submit();
});*/

/*document.getElementById("predictionForm").addEventListener("submit", function(event) {

  event.preventDefault(); // Prevent the form from submitting

  // Display the prediction text
  document.getElementById("predictionResult").style.display = "block";

  // Hide the prediction text after 5 seconds
  setTimeout(function() {
    document.getElementById("predictionResult").style.display = "none";
    
    // Trigger the chatbot after displaying the prediction result
    // Load the chatbot script dynamically
    var script = document.createElement("script");
    script.src = "//code.tidio.co/4yzyk7vfal8uvcpc9vceblbinc53efy7.js";
    script.async = true;
    document.body.appendChild(script);
  }, ); // 5 seconds = 5000 milliseconds
});
function autoRefresh() {
  window.location = window.location.href;
}
setInterval('autoRefresh()', 5000);*/
































function validateForm() {
  var snoringRate = document.getElementById("snoringRate").value;
  var bodyTemperature = document.getElementById("bodyTemperature").value;
  var bloodOxygen = document.getElementById("bloodOxygen").value;
  var respirationRate = document.getElementById("respirationRate").value;
  var sleepingHours = document.getElementById("sleepingHours").value;
  var heartRate = document.getElementById("heartRate").value;
  var headache = document.getElementById("headAche").value;
  var workingHours = document.getElementById("workingHours").value;
  
  var valid = true;
  
  // Validation rules
  if (!/^(47(\.\d+)?|48(\.\d+)?|49(\.\d+)?|[5-8]\d(\.\d+)?|99(\.0)?)$/.test(snoringRate)) {
    document.getElementById("snoringRateError").innerHTML = "Please enter a valid Snoring Rate between 47 and 99 (including float values).";
    valid = false;
  } else {
    document.getElementById("snoringRateError").innerHTML = "";
  }
  if (!/^(85(\.\d+)?|86(\.\d+)?|87(\.\d+)?|[5-8]\d(\.\d+)?|99(\.0)?)$/.test(bodyTemperature)) {
    document.getElementById("bodyTemperatureError").innerHTML = "Please enter a valid Body Temperature between 85 and 99 (including float values).";
    valid = false;
  } else {
    document.getElementById("bodyTemperatureError").innerHTML = "";
  }
  if (!/^(82(\.\d+)?|83(\.\d+)?|84(\.\d+)?|[5-8]\d(\.\d+)?|97(\.0)?)$/.test(bloodOxygen)) {
    document.getElementById("bloodOxygenError").innerHTML = "Please enter a valid Blood Oxygen between 82 and 97 (including float values).";
    valid = false;
  } else {
    document.getElementById("bloodOxygenError").innerHTML = "";
  }
  if (!/^(1[6-9](\.\d+)?|2[0-9](\.\d+)?|29(\.0)?)$/.test(respirationRate)) {
    document.getElementById("respirationRateError").innerHTML = "Please enter a valid Respiration Rate between 82 and 97 (including float values).";
    valid = false;
  } else {
    document.getElementById("respirationRateError").innerHTML = "";
  }
  if (!/^(?:[0-9](?:\.\d+)?|\.\d+)$/.test(sleepingHours)) {
    document.getElementById("sleepingHoursError").innerHTML = "Please enter a valid Sleeping Hours between 0 and 9 (including float values).";
    valid = false;
  } else {
    document.getElementById("sleepingHoursError").innerHTML = "";
  }
  if (!/^(50(\.\d+)?|51(\.\d+)?|52(\.\d+)?|[5-8]\d(\.\d+)?|84(\.0)?)$/.test(heartRate)) {
    document.getElementById("heartRateError").innerHTML = "Please enter a valid Heart Rate between 50 and 84 (including float values).";
    valid = false;
  } else {
    document.getElementById("heartRateError").innerHTML = "";
  }
  if (!/^([01])$/.test(headAche)) {
    document.getElementById("headAcheError").innerHTML = "Please enter only 0 or 1(Yes-1 , No-0)";
    valid = false;
  } else {
    document.getElementById("headAcheError").innerHTML = "";
  }
  if (!/^(0?[4-9]|1[0-5])$/.test(workingHours)) {
    document.getElementById("workingHoursError").innerHTML = "Please enter valid Working Hours between 4 and 15 (without decimal values).";
    valid = false;
  } else {
    document.getElementById("workingHoursError").innerHTML = "";
  }
  return valid;
  }