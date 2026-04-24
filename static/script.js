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
            showStatus('Error: Human Canvas not found', 'error', 'humanStatus');
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
                showStatus('Error: ' + captcha.error, 'error', 'humanStatus');
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
                showStatus('Error: No maze image received', 'error', 'humanStatus');
                return;
            }

            // Load maze image
            humanMazeImage = new Image();
            humanMazeImage.onload = function() {
                console.log('Human maze image loaded, drawing...');
                console.log('Canvas size:', humanCanvas.width, 'x', humanCanvas.height);
                console.log('Image size:', humanMazeImage.width, 'x', humanMazeImage.height);
                drawHumanMaze();
                showStatus('New human captcha loaded! Draw a path from green to red.', 'info', 'humanStatus');
            };
            humanMazeImage.onerror = function(error) {
                console.error('Failed to load human maze image:', error);
                showStatus('Error loading human maze image', 'error', 'humanStatus');
            };
            humanMazeImage.src = captcha.maze_image;
        })
        .catch(function(error) {
            console.error('Error loading human captcha:', error);
            showStatus('Error loading human captcha. Please try again.', 'error', 'humanStatus');
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
                showStatus('Error: No maze image received', 'error', 'humanStatus');
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
    showStatus('Human path cleared. Start drawing from green square.', 'info', 'humanStatus');
}

// Verify human solution
function verifyHumanSolution() {
    if (!currentHumanCaptcha) {
        showStatus('Please load a human captcha first.', 'error', 'humanStatus');
        return;
    }

    if (humanPath.length < 2) {
        showStatus('Please draw a path from start to end. Current path: ' + humanPath.length + ' points', 'error', 'humanStatus');
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
            showStatus('✅ ' + data.message + ' (Confidence: ' + (data.confidence * 100).toFixed(1) + '%)', 'success', 'humanStatus');
            updateAnalytics();
            // Clear current captcha after successful verification
            currentHumanCaptcha = null;
            humanPath = [];
        } else {
            showStatus('❌ Error: ' + data.message + ' (Confidence: ' + (data.confidence * 100).toFixed(1) + '%)', 'error', 'humanStatus');
            updateAnalytics();
        }
    })
    .catch(function(error) {
        if (error.message.includes('expired') || error.message.includes('refresh')) {
            showStatus('⚠️ CAPTCHA expired. Loading new one...', 'warning', 'humanStatus');
            loadNewHumanCaptcha();
        } else {
            showStatus('Error verifying solution: ' + error.message, 'error', 'humanStatus');
        }
        updateAnalytics();
    });
}

// Simulate bot
function simulateBot() {
    if (!currentBotCaptcha) {
        showStatus('Please load a bot captcha first.', 'error', 'botStatus');
        return;
    }

    showStatus('Simulating bot behavior...', 'info', 'botStatus');

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

            showStatus('Bot simulation complete: ' + detected + ' detected (Confidence: ' + confidence + '%)', isHuman ? 'success' : 'error', 'botStatus');

            // Draw bot path if available
            if (result.path && result.path.length > 0) {
                drawBotPath(result.path);
            }
        } else if (result.error) {
            showStatus('Bot simulation failed: ' + result.error, 'error', 'botStatus');
        } else {
            showStatus('Bot simulation failed - unexpected response', 'error', 'botStatus');
        }

        updateAnalytics();
    })
    .catch(function(error) {
        showStatus('Error simulating bot. Please try again.', 'error', 'botStatus');
    });
}

// Update analytics with comprehensive display
function updateAnalytics() {
    fetch('/api/analytics')
        .then(function(response) {
            if (!response.ok) throw new Error('Failed to load analytics');
            return response.json();
        })
        .then(function(data) {
            if (!data) return;
            analyticsData = data;

            var total = data.total_attempts || 0;
            var humanDetected = data.human_detected || 0;
            var botDetected = data.bot_detected || 0;

            document.getElementById('totalAttempts').textContent = total;
            document.getElementById('botDetected').textContent = botDetected;

            var successRate = total > 0 ? ((humanDetected / total) * 100).toFixed(1) : 0;
            var successRateElement = document.getElementById('successRate');
            if (successRateElement) {
                successRateElement.textContent = successRate + '%';
            }

            var avgTimeElement = document.getElementById('avgSolveTime');
            if (avgTimeElement) {
                avgTimeElement.textContent = '15.5s';
            }

            var confidenceElement = document.getElementById('avgConfidence');
            if (confidenceElement) {
                confidenceElement.textContent = '95%';
            }

            var avgPathElement = document.getElementById('avgPathLength');
            if (avgPathElement) {
                avgPathElement.textContent = total > 0 ? Math.round(total * 0.5) : 0;
            }

            // Update charts
            if (performanceChart) {
                var failed = total - humanDetected - botDetected;
                performanceChart.data.datasets[0].data = [
                    humanDetected,
                    botDetected,
                    failed
                ];
                performanceChart.update();
            }

if (confidenceChart) {
                confidenceChart.data.datasets[0].data = [0, 0, botDetected];
                confidenceChart.update();
            }
        })
        .catch(function(error) {
            console.log('Analytics update error:', error);
        });
}

// Show status message
function showStatus(message, type, targetId) {
    var statusDiv = document.getElementById(targetId || 'botStatus');
    if (!statusDiv) return;

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
    // Performance chart (doughnut) - black & white theme
    var performanceCtx = document.getElementById('performanceChart');
    if (performanceCtx) {
        try {
            performanceChart = new Chart(performanceCtx.getContext('2d'), {
                type: 'doughnut',
                data: {
                    labels: ['Human', 'Bots', 'Failed'],
                    datasets: [{
                        data: [0, 0, 0],
                        backgroundColor: [
                            'rgba(0, 0, 0, 0.7)',
                            'rgba(100, 100, 100, 0.7)',
                            'rgba(200, 200, 200, 0.7)'
                        ],
                        borderColor: '#000',
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                font: { family: "'IBM Plex Mono', monospace", size: 10 },
                                color: '#000'
                            }
                        }
                    }
                }
            });
        } catch (e) { console.log('Performance chart error:', e); }
    }

    // Confidence distribution chart - black & white theme
    var confidenceCtx = document.getElementById('confidenceHistogram');
    if (confidenceCtx) {
        try {
            confidenceChart = new Chart(confidenceCtx.getContext('2d'), {
                type: 'bar',
                data: {
                    labels: ['Low', 'Medium', 'High'],
                    datasets: [{
                        label: 'Confidence',
                        data: [0, 0, 0],
                        backgroundColor: 'rgba(0, 0, 0, 0.6)',
                        borderColor: '#000',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: { color: 'rgba(0,0,0,0.1)' },
                            ticks: { font: { family: "'IBM Plex Mono', monospace", size: 10 } }
                        },
                        x: {
                            grid: { display: false },
                            ticks: { font: { family: "'IBM Plex Mono', monospace", size: 10 } }
                        }
                    }
                }
            });
        } catch (e) { console.log('Confidence chart error:', e); }
    }
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