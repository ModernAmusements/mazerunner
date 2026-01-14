This README is designed to get your **Interactive Maze Solver** up and running on your Mac Pro or MacBook Pro in less than 2 minutes. It includes every command you need to type into your terminal.

---

# 🚀 Mac Pro Interactive Maze Solver

A high-performance, real-time maze generator and solver built with Python and OpenCV. Featuring **Turbo A*** pathfinding, a **Human-style** repaint mode, and a **Control Dashboard**.

## 🛠️ Installation & Setup

Open your **Terminal** and run the following commands in order:

### 1. Create a Project Folder

```bash
mkdir maze_project
cd maze_project

```

### 2. Set up a Virtual Environment

It is best practice to keep your dependencies isolated.

```bash
python3 -m venv maze_env
source maze_env/bin/activate

```

### 3. Install Dependencies

We use the `opencv-python` library for the engine and `numpy` for the high-speed math.

```bash
pip install --upgrade pip
pip install opencv-python numpy

```


### 4. Create the Script
Create a file named `bot.py` and paste the "Humanized Maze Solver" code we developed into it.

```bash
touch bot.py
open -e bot.py  # This opens it in TextEdit; paste the code and save.

```

---

## 🏃 How to Run

Once you have saved the code into `bot.py`, run it with:

```bash
python bot.py

```

---

## 🎮 Control Dashboard Guide

The UI appears at the top of the window. You can interact with it using your mouse or keyboard:

| Action | UI Button | Keyboard Key | Description |
| --- | --- | --- | --- |
| **Regenerate** | `REGEN` | `R` | Wipes the board and creates a brand new maze. |
| **Solve (AI)** | `SOLVE` | `S` | Triggers the A* algorithm to find the mathematical path. |
| **Human Mode** | `HUMAN` | N/A | Repaints the path with "hand-drawn" jitter and ink effects. |
| **Exit** | N/A | `ESC` | Closes the window and stops the script. |

---

## 💡 Troubleshooting for Mac Users

* **Window not appearing?** Make sure you clicked the Python icon in your Dock. Sometimes macOS opens the window behind your terminal.
* **Infinite "Generating" Loop?** Ensure you are using the latest version of the code where `state["regen"]` is set to `False` immediately upon entry.
* **Python Version:** If `python` doesn't work, try `python3`.

---

### Next Step

Would you like me to show you how to bundle this into a **standalone `.app` file** so you can run it without ever opening the terminal again?