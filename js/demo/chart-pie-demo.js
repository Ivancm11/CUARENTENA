// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = 'Nunito', '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#858796';

// Pie Chart Example
var ctx = document.getElementById("myPieChart");
var testData = {"infected":[0,55,120,1000,10000,7700,6895,8000,6000,5500,3000,1090],"circulo":[120,55,77],"regions":{"cat":{
  "population":{"x":[[4,5,6,7],[7,4,3,2]],
                "y":[[4,5,6,7],[7,4,3,2]],   
                "state":[[0,0,1,1],[]]}}}}; 
var myPieChart = new Chart(ctx, {
  type: 'doughnut',
  data: {
    labels: ["Infected", "Recovered", "Uninfected"],
    datasets: [{
      data: testData["circulo"],
      backgroundColor: ['#4e73df', '#1cc88a', '#36b9cc'],
      hoverBackgroundColor: ['#2e59d9', '#17a673', '#2c9faf'],
      hoverBorderColor: "rgba(234, 236, 244, 1)",
    }],
  },
  options: {
    maintainAspectRatio: false,
    tooltips: {
      backgroundColor: "rgb(255,255,255)",
      bodyFontColor: "#858796",
      borderColor: '#dddfeb',
      borderWidth: 1,
      xPadding: 15,
      yPadding: 15,
      displayColors: false,
      caretPadding: 10,
    },
    legend: {
      display: false
    },
    cutoutPercentage: 80,
  },
});
