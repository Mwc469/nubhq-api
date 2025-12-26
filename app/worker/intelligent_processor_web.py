#!/usr/bin/env python3
"""
NubHQ Intelligent Processor - Web UI
=====================================
Web interface for video processing prompts.
Shows pending decisions and learns from choices.
"""

import os
import json
import sqlite3
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import threading
from queue import Queue, Empty

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import from main processor
from intelligent_processor import (
    Config, VideoAnalyzer, VideoAnalysis, PreferenceLearner, 
    VideoProcessor, UserDecision, DecisionType, PromptSystem
)

# ============================================================
# MODELS
# ============================================================

class PendingDecision(BaseModel):
    """A decision waiting for user input"""
    id: str
    video_path: str
    video_name: str
    decision_type: str
    title: str
    description: str
    choices: List[Dict[str, str]]
    recommendation: Optional[str]
    confidence: float
    video_info: Dict[str, Any]


class DecisionResponse(BaseModel):
    """User's response to a decision"""
    decision_id: str
    choice: str
    always_apply: bool = False


class ProcessingStatus(BaseModel):
    """Status of a video being processed"""
    video_name: str
    status: str  # pending, processing, completed, failed
    progress: int
    current_step: str
    decisions_made: Dict[str, str]
    output_paths: List[str]
    error: Optional[str]


# ============================================================
# PROMPT QUEUE MANAGER
# ============================================================

class PromptQueueManager:
    """Manages pending prompts and their responses"""
    
    def __init__(self, learner: PreferenceLearner):
        self.learner = learner
        self.pending: Dict[str, PendingDecision] = {}
        self.responses: Dict[str, Queue] = {}
        self.processing: Dict[str, ProcessingStatus] = {}
        self._lock = threading.Lock()
        self._id_counter = 0
    
    def add_prompt(self, video_path: str, analysis: VideoAnalysis, 
                   decision_type: DecisionType) -> str:
        """Add a new prompt and wait for response"""
        with self._lock:
            self._id_counter += 1
            decision_id = f"d{self._id_counter}"
        
        opts = PromptSystem.OPTIONS.get(decision_type, {})
        rec, confidence = self.learner.get_recommendation(decision_type.value, analysis)
        
        pending = PendingDecision(
            id=decision_id,
            video_path=video_path,
            video_name=Path(video_path).name,
            decision_type=decision_type.value,
            title=opts.get('title', decision_type.value),
            description=opts.get('description', ''),
            choices=[{'key': k, 'label': v} for k, v in opts.get('choices', [])],
            recommendation=rec,
            confidence=confidence,
            video_info={
                'width': analysis.width,
                'height': analysis.height,
                'duration': analysis.duration,
                'fps': analysis.fps,
                'has_audio': analysis.has_audio,
                'is_dark': analysis.is_dark,
                'is_overexposed': analysis.is_overexposed,
                'peak_audio_db': analysis.peak_audio_db,
                'avg_audio_db': analysis.avg_audio_db,
            }
        )
        
        with self._lock:
            self.pending[decision_id] = pending
            self.responses[decision_id] = Queue()
        
        return decision_id
    
    def get_pending(self) -> List[PendingDecision]:
        """Get all pending decisions"""
        with self._lock:
            return list(self.pending.values())
    
    def submit_response(self, response: DecisionResponse) -> bool:
        """Submit a response to a pending decision"""
        with self._lock:
            if response.decision_id not in self.pending:
                return False
            
            pending = self.pending.pop(response.decision_id)
            queue = self.responses.get(response.decision_id)
            
            if queue:
                queue.put((response.choice, response.always_apply))
            
            return True
    
    def wait_for_response(self, decision_id: str, timeout: float = 300) -> Optional[tuple]:
        """Wait for a response to a decision"""
        queue = self.responses.get(decision_id)
        if not queue:
            return None
        
        try:
            result = queue.get(timeout=timeout)
            with self._lock:
                self.responses.pop(decision_id, None)
            return result
        except Empty:
            with self._lock:
                self.pending.pop(decision_id, None)
                self.responses.pop(decision_id, None)
            return None
    
    def update_status(self, video_name: str, status: ProcessingStatus):
        """Update processing status"""
        with self._lock:
            self.processing[video_name] = status
    
    def get_status(self) -> List[ProcessingStatus]:
        """Get all processing statuses"""
        with self._lock:
            return list(self.processing.values())


# ============================================================
# WEB APP
# ============================================================

app = FastAPI(title="NubHQ Intelligent Processor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
learner: Optional[PreferenceLearner] = None
queue_manager: Optional[PromptQueueManager] = None


@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    global learner, queue_manager
    Config.ensure_dirs()
    learner = PreferenceLearner(Config.DB_PATH)
    queue_manager = PromptQueueManager(learner)


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main UI"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🦭 NubHQ Video Processor</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        :root {
            --nub-primary: #FF6B35;
            --nub-secondary: #F7C59F;
            --nub-dark: #1a1a2e;
            --nub-surface: #16213e;
            --nub-text: #eee;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: var(--nub-dark);
            color: var(--nub-text);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container { max-width: 800px; margin: 0 auto; }
        
        header {
            text-align: center;
            padding: 40px 0;
        }
        
        header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        
        header p {
            opacity: 0.7;
            font-size: 1.1rem;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .stat-card {
            background: var(--nub-surface);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: 2px solid transparent;
        }
        
        .stat-card.pending { border-color: var(--nub-primary); }
        
        .stat-value {
            font-size: 2.5rem;
            font-weight: bold;
            color: var(--nub-primary);
        }
        
        .stat-label {
            opacity: 0.7;
            margin-top: 5px;
        }
        
        .decisions {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .decision-card {
            background: var(--nub-surface);
            border-radius: 16px;
            padding: 24px;
            border: 2px solid var(--nub-primary);
            animation: slideIn 0.3s ease;
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .decision-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 16px;
        }
        
        .decision-title {
            font-size: 1.3rem;
            font-weight: bold;
        }
        
        .decision-video {
            font-size: 0.9rem;
            opacity: 0.7;
            margin-top: 4px;
        }
        
        .video-info {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            margin: 12px 0;
            padding: 12px;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            font-size: 0.85rem;
        }
        
        .video-info span {
            background: rgba(255,255,255,0.1);
            padding: 4px 8px;
            border-radius: 4px;
        }
        
        .description {
            margin-bottom: 20px;
            opacity: 0.8;
        }
        
        .choices {
            display: grid;
            gap: 10px;
        }
        
        .choice-btn {
            background: rgba(255,255,255,0.05);
            border: 2px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 16px 20px;
            color: var(--nub-text);
            cursor: pointer;
            text-align: left;
            font-size: 1rem;
            transition: all 0.2s;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .choice-btn:hover {
            background: rgba(255,107,53,0.2);
            border-color: var(--nub-primary);
            transform: translateX(4px);
        }
        
        .choice-btn.recommended {
            border-color: var(--nub-primary);
            background: rgba(255,107,53,0.1);
        }
        
        .choice-btn.recommended::after {
            content: '← recommended';
            font-size: 0.8rem;
            opacity: 0.7;
        }
        
        .always-checkbox {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px solid rgba(255,255,255,0.1);
        }
        
        .always-checkbox input {
            width: 20px;
            height: 20px;
            accent-color: var(--nub-primary);
        }
        
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            opacity: 0.6;
        }
        
        .empty-state .walrus {
            font-size: 4rem;
            margin-bottom: 20px;
        }
        
        .processing-list {
            margin-top: 40px;
        }
        
        .processing-item {
            background: var(--nub-surface);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .progress-bar {
            width: 100px;
            height: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: var(--nub-primary);
            transition: width 0.3s;
        }
        
        .status-badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
        }
        
        .status-badge.completed { background: #2ecc71; }
        .status-badge.processing { background: var(--nub-primary); }
        .status-badge.failed { background: #e74c3c; }
        .status-badge.pending { background: #f39c12; }
        
        .learning-section {
            margin-top: 40px;
            background: var(--nub-surface);
            border-radius: 16px;
            padding: 24px;
        }
        
        .learning-section h2 {
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .learning-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        
        .learning-item:last-child { border-bottom: none; }
        
        .confidence-bar {
            width: 80px;
            height: 6px;
            background: rgba(255,255,255,0.1);
            border-radius: 3px;
            overflow: hidden;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(to right, #f39c12, #2ecc71);
        }
        
        .auto-badge {
            background: #2ecc71;
            color: #000;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🦭 NubHQ Video Processor</h1>
            <p>The walrus learns your preferences over time</p>
        </header>
        
        <div class="stats">
            <div class="stat-card pending">
                <div class="stat-value" id="pending-count">0</div>
                <div class="stat-label">Pending Decisions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="total-decisions">0</div>
                <div class="stat-label">Decisions Made</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="auto-count">0</div>
                <div class="stat-label">Auto-Applied</div>
            </div>
        </div>
        
        <div class="decisions" id="decisions">
            <div class="empty-state">
                <div class="walrus">🦭</div>
                <p>No pending decisions</p>
                <p>Drop videos into the input folder to start processing</p>
            </div>
        </div>
        
        <div class="processing-list" id="processing">
            <!-- Processing items will be added here -->
        </div>
        
        <div class="learning-section" id="learning">
            <h2>🧠 Learning Progress</h2>
            <div id="learning-items">
                <!-- Learning items will be added here -->
            </div>
        </div>
    </div>
    
    <script>
        // Poll for updates
        async function fetchUpdates() {
            try {
                // Get pending decisions
                const pendingRes = await fetch('/api/pending');
                const pending = await pendingRes.json();
                
                // Get learning stats
                const statsRes = await fetch('/api/stats');
                const stats = await statsRes.json();
                
                // Get processing status
                const statusRes = await fetch('/api/status');
                const status = await statusRes.json();
                
                updateUI(pending, stats, status);
            } catch (e) {
                console.error('Failed to fetch updates:', e);
            }
        }
        
        function updateUI(pending, stats, status) {
            // Update stats
            document.getElementById('pending-count').textContent = pending.length;
            document.getElementById('total-decisions').textContent = stats.total_decisions;
            
            const autoCount = Object.values(stats.preferences || {})
                .filter(p => p.auto_enabled).length;
            document.getElementById('auto-count').textContent = autoCount;
            
            // Update decisions
            const decisionsEl = document.getElementById('decisions');
            
            if (pending.length === 0) {
                decisionsEl.innerHTML = `
                    <div class="empty-state">
                        <div class="walrus">🦭</div>
                        <p>No pending decisions</p>
                        <p>Drop videos into the input folder to start processing</p>
                    </div>
                `;
            } else {
                decisionsEl.innerHTML = pending.map(d => `
                    <div class="decision-card" data-id="${d.id}">
                        <div class="decision-header">
                            <div>
                                <div class="decision-title">${d.title}</div>
                                <div class="decision-video">📁 ${d.video_name}</div>
                            </div>
                        </div>
                        
                        <div class="video-info">
                            <span>📐 ${d.video_info.width}x${d.video_info.height}</span>
                            <span>⏱️ ${d.video_info.duration.toFixed(1)}s</span>
                            <span>🎬 ${d.video_info.fps.toFixed(1)}fps</span>
                            ${d.video_info.has_audio ? '<span>🔊 Audio</span>' : '<span>🔇 No Audio</span>'}
                            ${d.video_info.is_dark ? '<span>🌑 Dark</span>' : ''}
                            ${d.video_info.is_overexposed ? '<span>☀️ Bright</span>' : ''}
                        </div>
                        
                        <p class="description">${d.description}</p>
                        
                        <div class="choices">
                            ${d.choices.map(c => `
                                <button class="choice-btn ${c.key === d.recommendation ? 'recommended' : ''}"
                                        onclick="submitChoice('${d.id}', '${c.key}')">
                                    ${c.label}
                                </button>
                            `).join('')}
                        </div>
                        
                        <label class="always-checkbox">
                            <input type="checkbox" id="always-${d.id}">
                            <span>Always use my choice for similar videos</span>
                        </label>
                    </div>
                `).join('');
            }
            
            // Update learning section
            const learningItems = document.getElementById('learning-items');
            const prefs = Object.entries(stats.preferences || {});
            
            if (prefs.length === 0) {
                learningItems.innerHTML = '<p style="opacity: 0.6">No preferences learned yet. Make some decisions!</p>';
            } else {
                learningItems.innerHTML = prefs.map(([type, info]) => `
                    <div class="learning-item">
                        <div>
                            <strong>${type}</strong>
                            <span style="opacity: 0.6; margin-left: 10px">${info.samples} samples</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 12px">
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${info.confidence * 100}%"></div>
                            </div>
                            ${info.auto_enabled ? '<span class="auto-badge">AUTO</span>' : `<span>${Math.round(info.confidence * 100)}%</span>`}
                        </div>
                    </div>
                `).join('');
            }
            
            // Update processing
            const processingEl = document.getElementById('processing');
            if (status.length > 0) {
                processingEl.innerHTML = '<h3 style="margin-bottom: 16px">📹 Processing</h3>' + 
                    status.map(s => `
                        <div class="processing-item">
                            <div>
                                <strong>${s.video_name}</strong>
                                <div style="opacity: 0.6; font-size: 0.9rem">${s.current_step}</div>
                            </div>
                            <div style="display: flex; align-items: center; gap: 16px">
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: ${s.progress}%"></div>
                                </div>
                                <span class="status-badge ${s.status}">${s.status}</span>
                            </div>
                        </div>
                    `).join('');
            } else {
                processingEl.innerHTML = '';
            }
        }
        
        async function submitChoice(decisionId, choice) {
            const alwaysCheckbox = document.getElementById(`always-${decisionId}`);
            const alwaysApply = alwaysCheckbox ? alwaysCheckbox.checked : false;
            
            try {
                const res = await fetch('/api/respond', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        decision_id: decisionId,
                        choice: choice,
                        always_apply: alwaysApply
                    })
                });
                
                if (res.ok) {
                    // Remove the card with animation
                    const card = document.querySelector(`[data-id="${decisionId}"]`);
                    if (card) {
                        card.style.opacity = '0';
                        card.style.transform = 'translateX(100px)';
                        setTimeout(() => fetchUpdates(), 300);
                    }
                }
            } catch (e) {
                console.error('Failed to submit choice:', e);
            }
        }
        
        // Initial fetch and polling
        fetchUpdates();
        setInterval(fetchUpdates, 2000);
    </script>
</body>
</html>
    """


@app.get("/api/pending")
async def get_pending():
    """Get all pending decisions"""
    return queue_manager.get_pending()


@app.post("/api/respond")
async def submit_response(response: DecisionResponse):
    """Submit a response to a decision"""
    success = queue_manager.submit_response(response)
    if not success:
        raise HTTPException(status_code=404, detail="Decision not found")
    return {"success": True}


@app.get("/api/stats")
async def get_stats():
    """Get learning statistics"""
    return learner.get_stats()


@app.get("/api/status")
async def get_status():
    """Get processing status"""
    return queue_manager.get_status()


# ============================================================
# PROCESSOR WITH WEB PROMPTS
# ============================================================

class WebVideoProcessor(VideoProcessor):
    """Video processor that uses web UI for prompts"""
    
    def __init__(self, learner: PreferenceLearner, queue_manager: PromptQueueManager):
        super().__init__(learner)
        self.queue_manager = queue_manager
    
    def _prompt_user(self, video_path: str, analysis: VideoAnalysis, 
                     decision_type: DecisionType) -> tuple:
        """Prompt user via web UI"""
        decision_id = self.queue_manager.add_prompt(video_path, analysis, decision_type)
        
        # Wait for response (5 minute timeout)
        result = self.queue_manager.wait_for_response(decision_id, timeout=300)
        
        if result is None:
            # Timeout - use recommendation or default
            rec, _ = self.learner.get_recommendation(decision_type.value, analysis)
            return rec or 'default', False
        
        return result


# ============================================================
# MAIN
# ============================================================

def run_server():
    """Run the web server"""
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")


if __name__ == '__main__':
    run_server()
