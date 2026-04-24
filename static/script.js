// Global variables
var currentHumanCaptcha = null;
var currentBotCaptcha = null;
var humanPath = [];
var botPath = [];
var isHumanDrawing = false;
var isBotDrawing = false;
var humanMazeImage = null;
var botMazeImage = null;
var analyticsData = null;
var performanceChart = null;
var learningChart = null;
var pathLengthChart = null;
var confidenceChart = null;
var hourlyChart = null;

// Canvas setup - will be initialized after DOM loads
var humanCanvas = null;
var botCanvas = null;
var humanCtx = null;
var botCtx = null;

// Load new human captcha
function loadNewHumanCaptcha() {
    console.log('Loading new human captcha...');

    // Ensure canvases are ready
    if (!humanCanvas || !humanCtx) {
        console.error('Human canvas not initialized, re-initializing...');
        humanCanvas = document.getElementById('humanMazeCanvas');
        if (humanCanvas) {
            humanCtx = humanCanvas.getContext('2d');
            console.log('Human canvas re-initialized successfully');
        } else {
            console.error('Cannot find human maze canvas element!');
            showStatus('Error: Human Canvas not found', 'error');
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

            console.log('=== HUMAN CAPTCHA LOADED ===');
            console.log('Captcha ID:', captcha.captcha_id);
            console.log('Captcha start:', captcha.start);
            console.log('Captcha end:', captcha.end);
            console.log('Maze image length:', captcha.maze_image ? captcha.maze_image.length : 'undefined');

            currentHumanCaptcha = captcha;
            // Convert maze coordinates to canvas coordinates for drawing indicators
            if (currentHumanCaptcha.start && currentHumanCaptcha.end) {
                currentHumanCaptcha.canvas_start = [
                    currentHumanCaptcha.start[1] * 20 + 10,  // col to x with center offset
                    currentHumanCaptcha.start[0] * 20 + 10   // row to y with center offset
                ];
                currentHumanCaptcha.canvas_end = [
                    currentHumanCaptcha.end[1] * 20 + 10,    // col to x with center offset
                    currentHumanCaptcha.end[0] * 20 + 10     // row to y with center offset
                ];
                console.log('Canvas start calculated:', currentHumanCaptcha.canvas_start);
                console.log('Canvas end calculated:', currentHumanCaptcha.canvas_end);
            }
            humanPath = [];
            isHumanDrawing = false;
            console.log('Human path reset');

            if (!captcha.maze_image) {
                console.error('No maze_image in captcha response');
                showStatus('Error: No maze image received', 'error');
                return;
            }

            // Load maze image
            humanMazeImage = new Image();
            humanMazeImage.onload = function() {
                console.log('Human maze image loaded, drawing...');
                console.log('Canvas size:', humanCanvas.width, 'x', humanCanvas.height);
                console.log('Image size:', humanMazeImage.width, 'x', humanMazeImage.height);
                drawHumanMaze();
                showStatus('New human captcha loaded! Draw a path from green to red.', 'info');
            };
            humanMazeImage.onerror = function(error) {
                console.error('Failed to load human maze image:', error);
                showStatus('Error loading human maze image', 'error');
            };
            humanMazeImage.src = captcha.maze_image;
        })
        .catch(function(error) {
            console.error('Error loading human captcha:', error);
            showStatus('Error loading human captcha. Please try again.', 'error');
        });
}

// Load new bot captcha (same maze structure but with different path)
function loadNewBotCaptcha() {
    console.log('Loading new bot captcha...');

    // Ensure canvases are ready
    if (!botCanvas || !botCtx) {
        console.error('Bot canvas not initialized, re-initializing...');
        botCanvas = document.getElementById('botMazeCanvas');
        if (botCanvas) {
            botCtx = botCanvas.getContext('2d');
            console.log('Bot canvas re-initialized successfully');
        } else {
            console.error('Cannot find bot maze canvas element!');
            showStatus('Error: Bot Canvas not found', 'error');
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

            console.log('=== BOT CAPTCHA LOADED ===');
            console.log('Captcha ID:', captcha.captcha_id);
            console.log('Captcha start:', captcha.start);
            console.log('Captcha end:', captcha.end);

            currentBotCaptcha = captcha;
            // Convert maze coordinates to canvas coordinates for drawing indicators
            if (currentBotCaptcha.start && currentBotCaptcha.end) {
                currentBotCaptcha.canvas_start = [
                    currentBotCaptcha.start[1] * 20 + 10,  // col to x with center offset
                    currentBotCaptcha.start[0] * 20 + 10   // row to y with center offset
                ];
                currentBotCaptcha.canvas_end = [
                    currentBotCaptcha.end[1] * 20 + 10,    // col to x with center offset
                    currentBotCaptcha.end[0] * 20 + 10     // row to y with center offset
                ];
            }
            botPath = [];
            isBotDrawing = false;
            console.log('Bot path reset');

            if (!captcha.maze_image) {
                console.error('No maze_image in captcha response');
                showStatus('Error: No maze image received', 'error');
                return;
            }

            // Load maze image
            botMazeImage = new Image();
            botMazeImage.onload = function() {
                console.log('Bot maze image loaded, drawing...');
                drawBotMaze();
                showStatus('New bot captcha loaded! Bot simulation ready.', 'info');
            };
            botMazeImage.onerror = function(error) {
                console.error('Failed to load bot maze image:', error);
                showStatus('Error loading bot maze image', 'error');
            };
            botMazeImage.src = captcha.maze_image;
        })
        .catch(function(error) {
            console.error('Error loading bot captcha:', error);
            showStatus('Error loading bot captcha. Please try again.', 'error');
        });
}

// Draw human maze
function drawHumanMaze() {
    if (!humanCtx) {
        console.error('Human canvas context is null, cannot draw');
        return;
    }

    // Always clear first
    humanCtx.clearRect(0, 0, humanCanvas.width, humanCanvas.height);

    if (!humanMazeImage) {
        // Draw placeholder test pattern
        humanCtx.fillStyle = '#333333';
        humanCtx.fillRect(0, 0, humanCanvas.width, humanCanvas.height);
        humanCtx.fillStyle = '#ffffff';
        humanCtx.font = '16px Arial';
        humanCtx.fillText('Loading maze...', 150, 200);
        return;
    }

    // Draw maze image first (as background)
    try {
        humanCtx.drawImage(humanMazeImage, 0, 0);
    } catch (error) {
        humanCtx.fillStyle = '#000000';
        humanCtx.fillRect(0, 0, humanCanvas.width, humanCanvas.height);
        humanCtx.fillStyle = '#ffffff';
        humanCtx.font = '12px Arial';
        humanCtx.fillText('Maze image failed', 150, 200);
    }

    // Draw human path on top of maze
    if (humanPath.length > 0) {
        humanCtx.strokeStyle = '#0066ff';
        humanCtx.lineWidth = 4;
        humanCtx.lineCap = 'round';
        humanCtx.lineJoin = 'round';

        humanCtx.beginPath();
        humanCtx.moveTo(humanPath[0].x, humanPath[0].y);

        for (var i = 1; i < humanPath.length; i++) {
            humanCtx.lineTo(humanPath[i].x, humanPath[i].y);
        }

        humanCtx.stroke();
    }
}

// Draw bot maze
function drawBotMaze() {
    if (!botCtx) {
        console.error('Bot canvas context is null, cannot draw');
        return;
    }

    // Always clear first
    botCtx.clearRect(0, 0, botCanvas.width, botCanvas.height);

    if (!botMazeImage) {
        // Draw placeholder test pattern
        botCtx.fillStyle = '#333333';
        botCtx.fillRect(0, 0, botCanvas.width, botCanvas.height);
        botCtx.fillStyle = '#ffffff';
        botCtx.font = '16px Arial';
        botCtx.fillText('Loading maze...', 150, 200);
        return;
    }

    // Draw maze image first (as background)
    try {
        botCtx.drawImage(botMazeImage, 0, 0);
    } catch (error) {
        botCtx.fillStyle = '#000000';
        botCtx.fillRect(0, 0, botCanvas.width, botCanvas.height);
        botCtx.fillStyle = '#ffffff';
        botCtx.font = '12px Arial';
        botCtx.fillText('Maze image failed', 150, 200);
    }
}

// Draw bot path from simulation results
function drawBotPath(path) {
    if (!botCtx || !path || path.length === 0) return;

    // Redraw maze first
    drawBotMaze();

    // Convert maze coordinates to canvas coordinates and draw path
    botCtx.strokeStyle = '#ff0000';
    botCtx.lineWidth = 4;
    botCtx.lineCap = 'round';
    botCtx.lineJoin = 'round';
    botCtx.setLineDash([5, 5]);

    botCtx.beginPath();
    for (var i = 0; i < path.length; i++) {
        var x = path[i][1] * 20 + 10;
        var y = path[i][0] * 20 + 10;
        if (i === 0) {
            botCtx.moveTo(x, y);
        } else {
            botCtx.lineTo(x, y);
        }
    }
    botCtx.stroke();
    botCtx.setLineDash([]);
}

// Initialize canvas events - call this after canvases are ready
function initializeCanvasEvents() {
    if (!humanCanvas || !humanCtx) {
        console.error('Cannot initialize human canvas events - canvas or ctx is null');
        return;
    }

    if (!botCanvas || !botCtx) {
        console.error('Cannot initialize bot canvas events - canvas or ctx is null');
        return;
    }

    // Remove existing listeners to avoid duplicates
    humanCanvas.removeEventListener('mousedown', handleHumanMouseDown);
    humanCanvas.removeEventListener('mousemove', handleHumanMouseMove);
    humanCanvas.removeEventListener('mouseup', handleHumanMouseUp);
    humanCanvas.removeEventListener('mouseleave', handleHumanMouseLeave);

    botCanvas.removeEventListener('mousedown', handleBotMouseDown);
    botCanvas.removeEventListener('mousemove', handleBotMouseMove);
    botCanvas.removeEventListener('mouseup', handleBotMouseUp);
    botCanvas.removeEventListener('mouseleave', handleBotMouseLeave);

    // Add event listeners for human
    humanCanvas.addEventListener('mousedown', handleHumanMouseDown);
    humanCanvas.addEventListener('mousemove', handleHumanMouseMove);
    humanCanvas.addEventListener('mouseup', handleHumanMouseUp);
    humanCanvas.addEventListener('mouseleave', handleHumanMouseLeave);

    // Add event listeners for bot
    botCanvas.addEventListener('mousedown', handleBotMouseDown);
    botCanvas.addEventListener('mousemove', handleBotMouseMove);
    botCanvas.addEventListener('mouseup', handleBotMouseUp);
    botCanvas.addEventListener('mouseleave', handleBotMouseLeave);
}

// Mouse event handlers for human
function handleHumanMouseDown(e) {
    if (!currentHumanCaptcha) {
        return;
    }

    if (!humanCanvas || !humanCtx) {
        console.error('Human canvas not available for mouse events');
        return;
    }

    var rect = humanCanvas.getBoundingClientRect();
    var x = Math.floor(e.clientX - rect.left);
    var y = Math.floor(e.clientY - rect.top);

    // Clear previous path and start fresh
    humanPath = [{x: x, y: y}];
    isHumanDrawing = true;
    drawHumanMaze();

    // Start tracking mouse data for this captcha
    fetch('/api/track', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        credentials: 'same-origin',
        body: JSON.stringify({
            captcha_id: currentHumanCaptcha.captcha_id,
            x: x,
            y: y,
            timestamp: Date.now() / 1000,
            event: 'mousedown'
        })
    }).catch(function(error) {
        // Optionally log error
    });
}

function handleHumanMouseMove(e) {
    if (!isHumanDrawing) {
        return;
    }

    if (!humanCanvas || !humanCtx) {
        console.error('Human canvas not available for mouse move');
        return;
    }

    var rect = humanCanvas.getBoundingClientRect();
    var x = Math.floor(e.clientX - rect.left);
    var y = Math.floor(e.clientY - rect.top);

    if (humanPath.length === 0) {
        return;
    }

    var lastPoint = humanPath[humanPath.length - 1];
    if (Math.abs(x - lastPoint.x) > 3 || Math.abs(y - lastPoint.y) > 3) {
        humanPath.push({x: x, y: y});
        drawHumanMaze();

        // Track mouse movement during drawing
        if (currentHumanCaptcha && currentHumanCaptcha.captcha_id) {
            fetch('/api/track', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                credentials: 'same-origin',
                body: JSON.stringify({
                    captcha_id: currentHumanCaptcha.captcha_id,
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

function handleHumanMouseUp() {
    isHumanDrawing = false;

    // Track mouse up event
    if (currentHumanCaptcha && currentHumanCaptcha.captcha_id && humanPath.length > 0) {
        fetch('/api/track', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            credentials: 'same-origin',
            body: JSON.stringify({
                captcha_id: currentHumanCaptcha.captcha_id,
                x: humanPath[humanPath.length - 1].x,
                y: humanPath[humanPath.length - 1].y,
                timestamp: Date.now() / 1000,
                event: 'mouseup'
            })
        }).catch(function(error) {
            // Optionally log error
        });
    }
}

function handleHumanMouseLeave() {
    isHumanDrawing = false;
}

// Mouse event handlers for bot
function handleBotMouseDown(e) {
    if (!currentBotCaptcha) {
        return;
    }

    if (!botCanvas || !botCtx) {
        console.error('Bot canvas not available for mouse events');
        return;
    }

    var rect = botCanvas.getBoundingClientRect();
    var x = Math.floor(e.clientX - rect.left);
    var y = Math.floor(e.clientY - rect.top);

    // Clear previous path and start fresh
    botPath = [{x: x, y: y}];
    isBotDrawing = true;
    drawBotMaze();

    // For bot, we don't track mouse movement here - it's handled in the bot simulation
}

function handleBotMouseMove(e) {
    if (!isBotDrawing) {
        return;
    }

    if (!botCanvas || !botCtx) {
        console.error('Bot canvas not available for mouse move');
        return;
    }

    var rect = botCanvas.getBoundingClientRect();
    var x = Math.floor(e.clientX - rect.left);
    var y = Math.floor(e.clientY - rect.top);

    if (botPath.length === 0) {
        return;
    }

    var lastPoint = botPath[botPath.length - 1];
    if (Math.abs(x - lastPoint.x) > 3 || Math.abs(y - lastPoint.y) > 3) {
        botPath.push({x: x, y: y});
        drawBotMaze();
    }
}

function handleBotMouseUp() {
    isBotDrawing = false;
}

function handleBotMouseLeave() {
    isBotDrawing = false;
}

// Clear human path
function clearHumanPath() {
    humanPath = [];
    isHumanDrawing = false;
    drawHumanMaze();
    showStatus('Human path cleared. Start drawing from green square.', 'info');
}

// Verify human solution
function verifyHumanSolution() {
    if (!currentHumanCaptcha) {
        showStatus('Please load a human captcha first.', 'error');
        return;
    }

    if (humanPath.length < 2) {
        showStatus('Please draw a path from start to end. Current path: ' + humanPath.length + ' points', 'error');
        return;
    }

    // Convert path to array format for server
    var pathArray = humanPath.map(function(point) {
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
            captcha_id: currentHumanCaptcha.captcha_id,
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
            currentHumanCaptcha = null;
            humanPath = [];
        } else {
            showStatus('❌ Error: ' + data.message + ' (Confidence: ' + (data.confidence * 100).toFixed(1) + '%)', 'error');
            updateAnalytics();
        }
    })
    .catch(function(error) {
        if (error.message.includes('expired') || error.message.includes('refresh')) {
            showStatus('⚠️ CAPTCHA expired. Loading new one...', 'warning');
            loadNewHumanCaptcha();
        } else {
            showStatus('Error verifying solution: ' + error.message, 'error');
        }
        updateAnalytics();
    });
}

// Simulate bot
function simulateBot() {
    if (!currentBotCaptcha) {
        showStatus('Please load a bot captcha first.', 'error');
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
            captcha_id: currentBotCaptcha.captcha_id
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

            // Draw bot path if available
            if (result.path && result.path.length > 0) {
                drawBotPath(result.path);
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
                if (data.recent_events && data.recent_events.length > 0) {
                    var eventsHtml = '';
                    data.recent_events.forEach(function(event) {
                        var timestamp = new Date(event.timestamp * 1000).toLocaleString();
                        var confidence = event.confidence ? (event.confidence * 100).toFixed(1) + '%' : 'N/A';
                        var typeClass = event.type === 'human_verified' ? 'human-event' : 
                                       event.type === 'bot_detected' ? 'bot-event' : 'system-event';
                        
                        eventsHtml += `
                            <div class="event-item ${typeClass}">
                                <span class="event-type">${event.type.replace('_', ' ').toUpperCase()}</span>
                                <span class="event-time">${timestamp}</span>
                                ${event.confidence ? `<span class="event-confidence">Confidence: ${confidence}</span>` : ''}
                                ${event.solve_time ? `<span class="event-confidence">Solve time: ${event.solve_time.toFixed(1)}s</span>` : ''}
                                ${event.reasons && event.reasons.length > 0 ? `<span class="event-confidence">Reason: ${event.reasons[0]}</span>` : ''}
                            </div>
                        `;
                    });
                    recentEventsDiv.innerHTML = eventsHtml;
                } else {
                    recentEventsDiv.innerHTML = '<p>No recent events available</p>';
                }
            }
        })
        .catch(function(error) {
            console.error('Error fetching analytics:', error);
            var recentEventsDiv = document.getElementById('recentEvents');
            if (recentEventsDiv) {
                recentEventsDiv.innerHTML = '<p>Error loading analytics data</p>';
            }
        });
}

// Show status message
function showStatus(message, type) {
    var statusDiv = document.getElementById('botStatus');
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
    humanCanvas = document.getElementById('humanMazeCanvas');
    botCanvas = document.getElementById('botMazeCanvas');
    
    if (humanCanvas) {
        humanCtx = humanCanvas.getContext('2d');
    }
    
    if (botCanvas) {
        botCtx = botCanvas.getContext('2d');
    }

    initializeCanvasEvents();

    initCharts();
    loadNewHumanCaptcha();
    loadNewBotCaptcha();
    updateAnalytics();

    setInterval(updateAnalytics, 10000);

    // setTimeout(testCanvas, 1000);
    // setTimeout(testPathDrawing, 2000);
};

// Test function to verify canvas works
function testCanvas() {
    if (!humanCanvas || !botCanvas || !humanCtx || !botCtx) {
        return;
    }

    try {
        humanCtx.fillStyle = '#00ff00';
        humanCtx.fillRect(5, 5, 30, 30);
        humanCtx.fillStyle = '#000000';
        humanCtx.font = '12px Arial';
        humanCtx.fillText('TEST', 10, 25);
        
        botCtx.fillStyle = '#ff0000';
        botCtx.fillRect(5, 5, 30, 30);
        botCtx.fillStyle = '#000000';
        botCtx.font = '12px Arial';
        botCtx.fillText('TEST', 10, 25);
    } catch (error) {}
}