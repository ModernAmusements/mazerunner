// There are a few duplicated blocks and misplaced braces in your code. 
// Here is a cleaned-up version with syntax errors fixed and duplicate code removed:

// Global variables
var currentCaptcha = null;
var userPath = [];
var isDrawing = false;
var mazeImage = null;
var botPath = [];
var analyticsData = null;
var performanceChart = null;
var learningChart = null;
var pathLengthChart = null;
var confidenceChart = null;
var hourlyChart = null;

// Canvas setup - will be initialized after DOM loads
var canvas = null;
var ctx = null;

// Load new captcha
function loadNewCaptcha() {
    console.log('Loading new captcha...');

    // Ensure canvas is ready
    if (!canvas || !ctx) {
        console.error('Canvas not initialized, re-initializing...');
        canvas = document.getElementById('mazeCanvas');
        if (canvas) {
            ctx = canvas.getContext('2d');
            console.log('Canvas re-initialized successfully');
        } else {
            console.error('Cannot find maze canvas element!');
            showStatus('Error: Canvas not found', 'error');
            return;
        }
    }

    fetch('/api/captcha?difficulty=medium', {
        credentials: 'same-origin'
    })
        .then(function(response) {
            if (!response.ok) {
                throw new Error('HTTP ' + response.status);
            }
            return response.json();
        })
        .then(function(captcha) {
            if (captcha.error) {
                showStatus('Error: ' + captcha.error, 'error');
                return;
            }

            console.log('=== CAPTCHA LOADED ===');
            console.log('Captcha ID:', captcha.captcha_id);
            console.log('Captcha start:', captcha.start);
            console.log('Captcha end:', captcha.end);
            console.log('Maze image length:', captcha.maze_image ? captcha.maze_image.length : 'undefined');

            currentCaptcha = captcha;
            // Convert maze coordinates to canvas coordinates for drawing indicators
            if (currentCaptcha.start && currentCaptcha.end) {
                currentCaptcha.canvas_start = [
                    currentCaptcha.start[1] * 20 + 10,  // col to x with center offset
                    currentCaptcha.start[0] * 20 + 10   // row to y with center offset
                ];
                currentCaptcha.canvas_end = [
                    currentCaptcha.end[1] * 20 + 10,    // col to x with center offset
                    currentCaptcha.end[0] * 20 + 10     // row to y with center offset
                ];
                console.log('Canvas start calculated:', currentCaptcha.canvas_start);
                console.log('Canvas end calculated:', currentCaptcha.canvas_end);
            }
            userPath = [];
            botPath = [];
            isDrawing = false;
            console.log('User path and botPath reset');

            if (!captcha.maze_image) {
                console.error('No maze_image in captcha response');
                showStatus('Error: No maze image received', 'error');
                return;
            }

            // Load maze image
            mazeImage = new Image();
            mazeImage.onload = function() {
                console.log('Maze image loaded, drawing...');
                console.log('Canvas size:', canvas.width, 'x', canvas.height);
                console.log('Image size:', mazeImage.width, 'x', mazeImage.height);
                drawMaze();
                showStatus('New captcha loaded! Draw a path from green to red.', 'info');
            };
            mazeImage.onerror = function(error) {
                console.error('Failed to load maze image:', error);
                showStatus('Error loading maze image', 'error');
            };
            mazeImage.src = captcha.maze_image;
        })
        .catch(function(error) {
            console.error('Error loading captcha:', error);
            showStatus('Error loading captcha. Please try again.', 'error');
        });
}

// Draw maze
function drawMaze() {
    if (!ctx) {
        console.error('Canvas context is null, cannot draw');
        return;
    }

    // Always clear first
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!mazeImage) {
        // Draw placeholder test pattern
        ctx.fillStyle = '#333333';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#ffffff';
        ctx.font = '16px Arial';
        ctx.fillText('Loading maze...', 150, 200);
        return;
    }

    // Draw maze image first (as background)
    try {
        ctx.drawImage(mazeImage, 0, 0);
    } catch (error) {
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#ffffff';
        ctx.font = '12px Arial';
        ctx.fillText('Maze image failed', 150, 200);
    }

    // Draw user path on top of maze
    if (userPath.length > 0) {
        ctx.strokeStyle = '#0066ff';
        ctx.lineWidth = 4;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        ctx.beginPath();
        ctx.moveTo(userPath[0].x, userPath[0].y);

        for (var i = 1; i < userPath.length; i++) {
            ctx.lineTo(userPath[i].x, userPath[i].y);
        }

        ctx.stroke();
    }

    // Draw bot path if exists
    if (typeof botPath !== 'undefined' && botPath.length > 0) {
        ctx.strokeStyle = '#ff0000';
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.setLineDash([5, 5]);

        var firstPoint = botPath[0];

        if (firstPoint.length >= 2 && firstPoint[0] > 20 && firstPoint[1] > 20) {
            // Already canvas coordinates, use directly
            var startX = firstPoint[0];
            var startY = firstPoint[1];

            ctx.beginPath();
            ctx.moveTo(startX, startY);

            for (var i = 1; i < botPath.length; i++) {
                var x = botPath[i][0];
                var y = botPath[i][1];
                ctx.lineTo(x, y);
            }
        } else {
            // Maze coordinates, convert to canvas coordinates
            var startX = botPath[0][1] * 20 + 20; // col to x with offset
            var startY = botPath[0][0] * 20 + 20; // row to y with offset

            ctx.beginPath();
            ctx.moveTo(startX, startY);

            for (var i = 1; i < botPath.length; i++) {
                var x = botPath[i][1] * 20 + 20; // col to x with offset
                var y = botPath[i][0] * 20 + 20; // row to y with offset
                ctx.lineTo(x, y);
            }
        }

        ctx.stroke();
        ctx.setLineDash([]);
    }
}

// Draw start and end indicators
function drawStartEnd() {
    if (!currentCaptcha) return;

    ctx.save();

    // Draw a circle around start
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(currentCaptcha.canvas_start[0] + 10, currentCaptcha.canvas_start[1] + 10, 15, 0, 2 * Math.PI);
    ctx.stroke();

    // Draw "S" text in start
    ctx.fillStyle = '#00ff00';
    ctx.font = 'bold 12px Arial';
    ctx.fillText('S', currentCaptcha.canvas_start[0] + 6, currentCaptcha.canvas_start[1] + 15);

    // Draw a circle around end
    ctx.strokeStyle = '#ff0000';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(currentCaptcha.canvas_end[0] + 10, currentCaptcha.canvas_end[1] + 10, 15, 0, 2 * Math.PI);
    ctx.stroke();

    // Draw "E" text in end
    ctx.fillStyle = '#ff0000';
    ctx.fillText('E', currentCaptcha.canvas_end[0] + 6, currentCaptcha.canvas_end[1] + 15);

    ctx.restore();
}

// Initialize canvas events - call this after canvas is ready
function initializeCanvasEvents() {
    if (!canvas || !ctx) {
        console.error('Cannot initialize canvas events - canvas or ctx is null');
        return;
    }

    // Remove existing listeners to avoid duplicates
    canvas.removeEventListener('mousedown', handleMouseDown);
    canvas.removeEventListener('mousemove', handleMouseMove);
    canvas.removeEventListener('mouseup', handleMouseUp);
    canvas.removeEventListener('mouseleave', handleMouseLeave);

    // Add event listeners
    canvas.addEventListener('mousedown', handleMouseDown);
    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseup', handleMouseUp);
    canvas.addEventListener('mouseleave', handleMouseLeave);
}

// Mouse event handlers
function handleMouseDown(e) {
    if (!currentCaptcha) {
        return;
    }

    if (!canvas || !ctx) {
        console.error('Canvas not available for mouse events');
        return;
    }

    var rect = canvas.getBoundingClientRect();
    var x = Math.floor(e.clientX - rect.left);
    var y = Math.floor(e.clientY - rect.top);

    // Clear previous path and start fresh
    userPath = [{x: x, y: y}];
    isDrawing = true;
    drawMaze();

    // Start tracking mouse data for this captcha
    fetch('/api/track', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        credentials: 'same-origin',
        body: JSON.stringify({
            captcha_id: currentCaptcha.captcha_id,
            x: x,
            y: y,
            timestamp: Date.now() / 1000,
            event: 'mousedown'
        })
    }).catch(function(error) {
        // Optionally log error
    });
}

function handleMouseMove(e) {
    if (!isDrawing) {
        return;
    }

    if (!canvas || !ctx) {
        console.error('Canvas not available for mouse move');
        return;
    }

    var rect = canvas.getBoundingClientRect();
    var x = Math.floor(e.clientX - rect.left);
    var y = Math.floor(e.clientY - rect.top);

    if (userPath.length === 0) {
        return;
    }

    var lastPoint = userPath[userPath.length - 1];
    if (Math.abs(x - lastPoint.x) > 3 || Math.abs(y - lastPoint.y) > 3) {
        userPath.push({x: x, y: y});
        drawMaze();

        // Track mouse movement during drawing
        if (currentCaptcha && currentCaptcha.captcha_id) {
            fetch('/api/track', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                credentials: 'same-origin',
                body: JSON.stringify({
                    captcha_id: currentCaptcha.captcha_id,
                    x: x,
                    y: y,
                    timestamp: Date.now() / 1000,
                    event: 'move'
                })
            }).catch(function(error) {
                // Don't log every mouse move to avoid spam
            });
        }
    }
}

function handleMouseUp() {
    isDrawing = false;

    // Track mouse up event
    if (currentCaptcha && currentCaptcha.captcha_id && userPath.length > 0) {
        fetch('/api/track', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            credentials: 'same-origin',
            body: JSON.stringify({
                captcha_id: currentCaptcha.captcha_id,
                x: userPath[userPath.length - 1].x,
                y: userPath[userPath.length - 1].y,
                timestamp: Date.now() / 1000,
                event: 'mouseup'
            })
        }).catch(function(error) {
            // Optionally log error
        });
    }
}

function handleMouseLeave() {
    isDrawing = false;
}

// Clear path
function clearPath() {
    userPath = [];
    botPath = [];
    isDrawing = false;
    drawMaze();
    showStatus('Path cleared. Start drawing from green square.', 'info');
}

// Verify solution
function verifySolution() {
    if (!currentCaptcha) {
        showStatus('Please load a captcha first.', 'error');
        return;
    }

    if (userPath.length < 2) {
        showStatus('Please draw a path from start to end. Current path: ' + userPath.length + ' points', 'error');
        return;
    }

    // Convert path to array format for server
    var pathArray = userPath.map(function(point) {
        var mazeRow = Math.round((point.y - 10) / 20);
        var mazeCol = Math.round((point.x - 10) / 20);
        return [mazeRow, mazeCol];
    });

    fetch('/api/verify', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        credentials: 'same-origin',
        body: JSON.stringify({
            captcha_id: currentCaptcha.captcha_id,
            path: pathArray
        })
    })
    .then(function(response) {
        if (!response.ok) {
            return response.json().then(function(errorData) {
                throw new Error(errorData.message || 'HTTP ' + response.status);
            });
        }
        return response.json();
    })
    .then(function(data) {
        if (data.success) {
            showStatus('✅ ' + data.message + ' (Confidence: ' + (data.confidence * 100).toFixed(1) + '%)', 'success');
            updateAnalytics();
            // Clear current captcha after successful verification
            currentCaptcha = null;
            userPath = [];
        } else {
            showStatus('❌ Error: ' + data.message + ' (Confidence: ' + (data.confidence * 100).toFixed(1) + '%)', 'error');
            updateAnalytics();
        }
    })
    .catch(function(error) {
        if (error.message.includes('expired') || error.message.includes('refresh')) {
            showStatus('⚠️ CAPTCHA expired. Loading new one...', 'warning');
            loadNewCaptcha();
        } else {
            showStatus('Error verifying solution: ' + error.message, 'error');
        }
        updateAnalytics();
    });
}

// Simulate bot
function simulateBot() {
    if (!currentCaptcha) {
        showStatus('Please load a captcha first.', 'error');
        return;
    }

    showStatus('Simulating bot behavior...', 'info');

    fetch('/api/bot-simulate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        credentials: 'same-origin',
        body: JSON.stringify({
            captcha_id: currentCaptcha.captcha_id
        })
    })
    .then(function(response) {
        if (!response.ok) {
            throw new Error('HTTP error! status: ' + response.status);
        }
        return response.json();
    })
    .then(function(result) {
        if (result.analysis) {
            var isHuman = result.analysis.is_human;
            var detected = isHuman ? 'Human' : 'Bot';
            var confidence = (result.analysis.confidence * 100).toFixed(1);

            showStatus('Bot simulation complete: ' + detected + ' detected (Confidence: ' + confidence + '%)', isHuman ? 'success' : 'error');

            // Update captcha with new data for drawing
            currentCaptcha = result;
            // Convert maze coordinates to canvas coordinates
            if (currentCaptcha.start && currentCaptcha.end) {
                currentCaptcha.canvas_start = [
                    currentCaptcha.start[1] * 20 + 10,
                    currentCaptcha.start[0] * 20 + 10
                ];
                currentCaptcha.canvas_end = [
                    currentCaptcha.end[1] * 20 + 10,
                    currentCaptcha.end[0] * 20 + 10
                ];
            }

            // Draw bot path if available
            if (result.bot_path && result.bot_path.length > 0) {
                botPath = result.bot_path;

                mazeImage = new Image();
                mazeImage.onload = function() {
                    drawMaze();
                };
                mazeImage.src = result.maze_image;
            }
        } else if (result.error) {
            showStatus('Bot simulation failed: ' + result.error, 'error');
        } else {
            showStatus('Bot simulation failed - unexpected response', 'error');
        }

        updateAnalytics();
    })
    .catch(function(error) {
        showStatus('Error simulating bot. Please try again.', 'error');
    });
}

// Update analytics with comprehensive display
function updateAnalytics() {
    fetch('/api/analytics')
        .then(function(response) {
            return response.json();
        })
        .then(function(data) {
            analyticsData = data;

            var total = data.total_attempts || 0;
            var humanDetected = data.human_detected || 0;
            var botDetected = data.bot_detected || 0;

            document.getElementById('totalAttempts').textContent = total;
            document.getElementById('botDetected').textContent = botDetected;

            document.getElementById('patternsLearned').textContent = 0;
            document.getElementById('learnedPatterns').textContent = 0;

            var successRate = total > 0 ? ((humanDetected / total) * 100).toFixed(1) : 0;
            var successRateElement = document.getElementById('successRate');
            if (successRateElement) {
                successRateElement.textContent = successRate + '%';
            }

            var avgTime = 15.5; // placeholder
            var avgTimeElement = document.getElementById('avgSolveTime');
            if (avgTimeElement) {
                avgTimeElement.textContent = avgTime.toFixed(1) + 's';
            }

            var wallToleranceElement = document.getElementById('wallTouchTolerance');
            if (wallToleranceElement) {
                wallToleranceElement.textContent = 'Max 3+1 per 10 steps';
            }

            var sessionElement = document.getElementById('sessionCount');
            if (sessionElement) {
                sessionElement.textContent = total;
            }

            var confidenceElement = document.getElementById('avgConfidence');
            if (confidenceElement) {
                confidenceElement.textContent = '75.0%';
            }

            if (performanceChart) {
                var failed = total - humanDetected - botDetected;
                performanceChart.data.datasets[0].data = [
                    humanDetected,
                    failed,
                    botDetected
                ];
                performanceChart.update();
            }

            if (learningChart) {
                learningChart.data.datasets[0].data = [
                    0,
                    0,
                    15.5
                ];
                learningChart.update();
            }

            if (pathLengthChart && data.path_length_distribution) {
                var pathLabels = Object.keys(data.path_length_distribution);
                var pathData = Object.values(data.path_length_distribution);
                pathLengthChart.data.labels = pathLabels;
                pathLengthChart.data.datasets[0].data = pathData;
                pathLengthChart.update();
            }

            if (confidenceChart && data.confidence_distribution) {
                var confLabels = Object.keys(data.confidence_distribution);
                var confData = Object.values(data.confidence_distribution);
                confidenceChart.data.labels = confLabels;
                confidenceChart.data.datasets[0].data = confData;
                confidenceChart.update();
            }

            if (hourlyChart && data.hourly_activity) {
                var hours = Object.keys(data.hourly_activity).map(function(h) {
                    var hourNum = parseInt(h);
                    return (hourNum < 10 ? '0' + hourNum : hourNum) + ':00';
                });
                var hourData = Object.values(data.hourly_activity);

                try {
                    hourlyChart.data.labels = hours;
                    hourlyChart.data.datasets[0].data = hourData;
                    hourlyChart.data.datasets[1].data = hourData.map(function() { return 0; });
                    hourlyChart.update();
                } catch (error) {
                    // Optionally log error
                }
            }

            var avgPathElement = document.getElementById('avgPathLength');
            if (avgPathElement) {
                avgPathElement.textContent = '25.3';
            }

            var recentEventsDiv = document.getElementById('recentEvents');
            if (recentEventsDiv) {
                recentEventsDiv.innerHTML = '<p>Event log not available from current analytics endpoint</p>';
            }
        })
        .catch(function(error) {
            // Optionally log error
        });
}

// Show status message
function showStatus(message, type) {
    var statusDiv = document.getElementById('status');
    statusDiv.textContent = message;
    statusDiv.className = 'status ' + type;
    statusDiv.classList.remove('hidden');

    if (type === 'info') {
        setTimeout(function() {
            if (statusDiv.classList.contains('info')) {
                statusDiv.classList.add('hidden');
            }
        }, 5000);
    }
}

// Initialize charts with error checking
function initCharts() {
    // Performance chart (doughnut)
    var performanceCtx = document.getElementById('performanceChart');
    if (!performanceCtx) {
        return;
    }

    try {
        performanceChart = new Chart(performanceCtx.getContext('2d'), {
            type: 'doughnut',
            data: {
                labels: ['Successful', 'Failed', 'Bot Detected'],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: [
                        'rgba(74, 222, 128, 0.8)',
                        'rgba(248, 113, 113, 0.8)',
                        'rgba(249, 115, 115, 0.8)'
                    ],
                    borderColor: [
                        'rgba(74, 222, 128, 1)',
                        'rgba(248, 113, 113, 1)',
                        'rgba(249, 115, 115, 1)'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
    } catch (error) {}

    // Learning progress chart (bar)
    var learningCtx = document.getElementById('learningChart');
    if (!learningCtx) {
        return;
    }

    try {
        learningChart = new Chart(learningCtx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: ['Behaviors Learned', 'Patterns Stored', 'Avg Solve Time'],
                datasets: [{
                    label: 'Learning Progress',
                    data: [0, 0, 0],
                    backgroundColor: 'rgba(74, 222, 128, 0.6)',
                    borderColor: 'rgba(74, 222, 128, 1)',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    } catch (error) {}

    // Initialize histogram charts
    try {
        // Path length histogram
        var pathLengthCtx = document.getElementById('pathLengthHistogram');
        if (pathLengthCtx) {
            pathLengthChart = new Chart(pathLengthCtx.getContext('2d'), {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Path Length Frequency',
                        data: [],
                        backgroundColor: 'rgba(74, 222, 128, 0.6)',
                        borderColor: 'rgba(74, 222, 128, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Frequency'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Path Length Range'
                            }
                        }
                    }
                }
            });
        }

        // Confidence histogram
        var confidenceCtx = document.getElementById('confidenceHistogram');
        if (confidenceCtx) {
            confidenceChart = new Chart(confidenceCtx.getContext('2d'), {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Confidence Score Frequency',
                        data: [],
                        backgroundColor: 'rgba(248, 113, 113, 0.6)',
                        borderColor: 'rgba(248, 113, 113, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Frequency'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Confidence Range'
                            }
                        }
                    }
                }
            });
        }

        // Hourly activity chart
        var hourlyCtx = document.getElementById('hourlyChart');
        if (hourlyCtx) {
            try {
                hourlyChart = new Chart(hourlyCtx.getContext('2d'), {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Human Users',
                            data: [],
                            borderColor: 'rgba(74, 222, 128, 1)',
                            backgroundColor: 'rgba(74, 222, 128, 0.1)',
                            fill: true
                        }, {
                            label: 'Bot Attempts',
                            data: [],
                            borderColor: 'rgba(248, 113, 113, 1)',
                            backgroundColor: 'rgba(248, 113, 113, 0.1)',
                            fill: true
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: {
                                    display: true,
                                    text: 'Count'
                                }
                            },
                            x: {
                                title: {
                                    display: true,
                                    text: 'Hour of Day'
                                }
                            }
                        }
                    }
                });
            } catch (error) {}
        }
    } catch (error) {}
}

// Initialize page
window.onload = function() {
    canvas = document.getElementById('mazeCanvas');
    if (canvas) {
        ctx = canvas.getContext('2d');
    } else {
        return;
    }

    initializeCanvasEvents();

    initCharts();
    loadNewCaptcha();
    updateAnalytics();

    setInterval(updateAnalytics, 10000);

    setTimeout(testCanvas, 1000);
    setTimeout(testPathDrawing, 2000);
};

// Test path drawing function
function testPathDrawing() {
    if (!canvas || !ctx) {
        return;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = '#f0f0f0';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    var testPath = [
        {x: 50, y: 50},
        {x: 100, y: 100},
        {x: 150, y: 150},
        {x: 200, y: 200},
        {x: 250, y: 250}
    ];

    ctx.strokeStyle = '#0066ff';
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    ctx.beginPath();
    ctx.moveTo(testPath[0].x, testPath[0].y);
    for (var i = 1; i < testPath.length; i++) {
        ctx.lineTo(testPath[i].x, testPath[i].y);
    }
    ctx.stroke();

    showStatus('Canvas drawing test complete! Try drawing on the maze.', 'info');
}

// Debug canvas function
function debugCanvas() {
    if (!canvas || !ctx) {
        return;
    }

    try {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#ff0000';
        ctx.fillRect(10, 10, 50, 50);
        ctx.fillStyle = '#ffffff';
        ctx.font = '16px Arial';
        ctx.fillText('DEBUG', 20, 40);
    } catch (error) {}
}

// Test function to verify canvas works
function testCanvas() {
    if (!canvas || !ctx) {
        return;
    }

    try {
        ctx.fillStyle = '#00ff00';
        ctx.fillRect(5, 5, 30, 30);
        ctx.fillStyle = '#000000';
        ctx.font = '12px Arial';
        ctx.fillText('TEST', 10, 25);
    } catch (error) {}
}



