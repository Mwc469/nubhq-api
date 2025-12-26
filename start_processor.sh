#!/bin/bash
#
# NubHQ Intelligent Video Processor
# ==================================
# Watches input folder, processes videos, learns preferences
#
# Usage:
#   ./start_processor.sh          # Start in terminal mode
#   ./start_processor.sh --web    # Start with web UI
#   ./start_processor.sh --setup  # First-time setup
#

set -e

# ============================================================
# CONFIGURATION
# ============================================================

# Default paths (can be overridden with environment variables)
export NUBHQ_INPUT="${NUBHQ_INPUT:-/Volumes/NUB_Workspace/input}"
export NUBHQ_PROCESSING="${NUBHQ_PROCESSING:-/Volumes/NUB_Workspace/processing}"
export NUBHQ_OUTPUT="${NUBHQ_OUTPUT:-/Volumes/NUB_Workspace/output}"
export NUBHQ_ARCHIVE="${NUBHQ_ARCHIVE:-/Volumes/NUB_Workspace/archive}"
export NUBHQ_DATA="${NUBHQ_DATA:-/Volumes/NUB_Workspace/.nubhq}"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_DIR="${SCRIPT_DIR}/backend/worker"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ============================================================
# FUNCTIONS
# ============================================================

print_header() {
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}     🦭 ${GREEN}NubHQ Intelligent Video Processor${NC}                   ${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

check_dependencies() {
    echo -e "${BLUE}Checking dependencies...${NC}"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python 3 not found${NC}"
        exit 1
    fi
    echo -e "  ${GREEN}✓${NC} Python 3 found"
    
    # Check FFmpeg
    if ! command -v ffmpeg &> /dev/null; then
        echo -e "${RED}❌ FFmpeg not found${NC}"
        echo "  Install with: brew install ffmpeg"
        exit 1
    fi
    echo -e "  ${GREEN}✓${NC} FFmpeg found"
    
    # Check FFprobe
    if ! command -v ffprobe &> /dev/null; then
        echo -e "${RED}❌ FFprobe not found${NC}"
        exit 1
    fi
    echo -e "  ${GREEN}✓${NC} FFprobe found"
}

setup_folders() {
    echo -e "${BLUE}Setting up folders...${NC}"
    
    mkdir -p "$NUBHQ_INPUT"
    mkdir -p "$NUBHQ_PROCESSING"
    mkdir -p "$NUBHQ_OUTPUT"
    mkdir -p "$NUBHQ_ARCHIVE"
    mkdir -p "$NUBHQ_DATA"
    
    echo -e "  ${GREEN}✓${NC} Input:      $NUBHQ_INPUT"
    echo -e "  ${GREEN}✓${NC} Processing: $NUBHQ_PROCESSING"
    echo -e "  ${GREEN}✓${NC} Output:     $NUBHQ_OUTPUT"
    echo -e "  ${GREEN}✓${NC} Archive:    $NUBHQ_ARCHIVE"
    echo -e "  ${GREEN}✓${NC} Data:       $NUBHQ_DATA"
}

install_python_deps() {
    echo -e "${BLUE}Installing Python dependencies...${NC}"
    
    pip3 install -q fastapi uvicorn pydantic 2>/dev/null || {
        echo -e "${YELLOW}  Installing with --user flag...${NC}"
        pip3 install --user fastapi uvicorn pydantic
    }
    
    echo -e "  ${GREEN}✓${NC} Python dependencies installed"
}

show_config() {
    echo ""
    echo -e "${YELLOW}Current Configuration:${NC}"
    echo -e "  📁 Input folder:    ${CYAN}$NUBHQ_INPUT${NC}"
    echo -e "  🔄 Processing:      ${CYAN}$NUBHQ_PROCESSING${NC}"
    echo -e "  📤 Output folder:   ${CYAN}$NUBHQ_OUTPUT${NC}"
    echo -e "  🗄️  Archive folder:  ${CYAN}$NUBHQ_ARCHIVE${NC}"
    echo -e "  📊 Data folder:     ${CYAN}$NUBHQ_DATA${NC}"
    echo ""
}

run_setup() {
    print_header
    echo -e "${YELLOW}First-time Setup${NC}"
    echo ""
    
    check_dependencies
    echo ""
    
    setup_folders
    echo ""
    
    install_python_deps
    echo ""
    
    show_config
    
    echo -e "${GREEN}✅ Setup complete!${NC}"
    echo ""
    echo "To start processing:"
    echo -e "  ${CYAN}./start_processor.sh${NC}       # Terminal mode"
    echo -e "  ${CYAN}./start_processor.sh --web${NC} # Web UI mode"
    echo ""
    echo "Drop videos into:"
    echo -e "  ${CYAN}$NUBHQ_INPUT${NC}"
    echo ""
}

run_terminal() {
    print_header
    show_config
    
    cd "$WORKER_DIR"
    python3 intelligent_processor.py
}

run_web() {
    print_header
    show_config
    
    echo -e "${GREEN}Starting web UI...${NC}"
    echo -e "Open ${CYAN}http://localhost:8765${NC} in your browser"
    echo ""
    
    cd "$WORKER_DIR"
    
    # Start web server in background
    python3 intelligent_processor_web.py &
    WEB_PID=$!
    
    # Wait a moment for server to start
    sleep 2
    
    # Open browser
    if command -v open &> /dev/null; then
        open "http://localhost:8765"
    fi
    
    # Start processor (will use web prompts)
    python3 intelligent_processor.py --web
    
    # Cleanup
    kill $WEB_PID 2>/dev/null || true
}

show_help() {
    print_header
    echo "Usage: $0 [option]"
    echo ""
    echo "Options:"
    echo "  (none)    Start in terminal mode (prompts in terminal)"
    echo "  --web     Start with web UI (prompts in browser)"
    echo "  --setup   First-time setup"
    echo "  --help    Show this help"
    echo ""
    echo "Environment Variables:"
    echo "  NUBHQ_INPUT       Input folder path"
    echo "  NUBHQ_OUTPUT      Output folder path"
    echo "  NUBHQ_ARCHIVE     Archive folder path"
    echo "  NUBHQ_PROCESSING  Processing folder path"
    echo "  NUBHQ_DATA        Data folder path"
    echo ""
}

# ============================================================
# MAIN
# ============================================================

case "${1:-}" in
    --setup)
        run_setup
        ;;
    --web)
        run_web
        ;;
    --help|-h)
        show_help
        ;;
    *)
        run_terminal
        ;;
esac
