document.addEventListener('DOMContentLoaded',() => {
     const chartElement = document.getElementById('chart');
     if (!chartElement) {
        console.error('Chart element not found');
        return;
     }

        const ctx = chartElement.getContext('2d');
        const gradient = ctx.createLinearGradient(0, -10, 0, 100);
        gradient.addColorStop(0, 'rgba(250, 0, 0, 1)');
        gradient.addColorStop(1, 'rgba(136, 255, 0, 1)');

        const forecastItems = document.querySelectorAll('.forecast-item');12

        const temps = [];
        const times = [];

        forecastItems.forEach(item => {
            const time = item.querySelector('.forecast-time').textContent;
            const temp = item.querySelector('.forecast-temperatureValue').textContent;
            const humidity = item.querySelector('.forecast-humidityValue').textContent;

            if(time && temp && hum) {
                times.push(time);
                temps.push(temp);
            }
        });

        //Ensure all the values are valid before using them

        if (temps.length === 0 || times.length === 0) {
            console.error('Temp and Time values ar missing');
            return;
        }

        new Chart(ctx, {
            type: 'line',
            data: {
                labels: times,
                datasets: [{
                    label: 'Temperature (°C)',
                    data: temps,
                    borderColor: gradient,
                    borderWidth: 2,
                    tension: 0.4,
                    pointRadius: 3,
                },
            ],
            },
            options: {
                Plugins: {
                    legend: {
                        display: false,
                    },
                },
                scales: {
                    x: {
                        display: false,
                        grid: {
                            drawOnChartArea: false,
                        },
                    },
                    y: {
                        display: false,
                        grid: {
                            drawOnChartArea: false,
                        },
                    },
                },
                Animation: {
                    duration: 750,
                },
            },
        });
    });

