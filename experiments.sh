#!/bin/bash

# ========== CONFIG ==========

EXPERIMENTS=(
  transformer_1t
  transformer_2t
  exact_1i_0t
  exact_1i_1t
)

# ========== SCRIPTS ==========

function experiment_command() {
  local name="$1"
  local device_id="$2"
  echo "source ~/.bashrc && conda activate icl && exec -a wac_${name} python main.py --train ${name} --device ${device_id}"
}

function python_command_display() {
  local name="$1"
  local device_id="$2"
  echo "python main.py --train ${name} --device ${device_id}"
}

function session_exists() {
  tmux has-session -t "$1" 2>/dev/null
}

ACTION=$1

if [ -z "$ACTION" ]; then
  echo "Usage: $0 [train|list|kill]"
  exit 1
fi

case "$ACTION" in

  train)
    for i in "${!EXPERIMENTS[@]}"; do
      EXP="${EXPERIMENTS[$i]}"
      DEVICE_ID="$i"
      if session_exists "$EXP"; then
        echo "Session '$EXP' already exists. Skipping."
      else
        CMD=$(experiment_command "$EXP" "$DEVICE_ID")
        tmux new-session -d -s "$EXP"
        tmux send-keys -t "$EXP" "$CMD" C-m
        echo "Started session '$EXP' with command: $(python_command_display "$EXP" "$DEVICE_ID")"
      fi
    done
    ;;

  list)
    while true; do
      echo "Available experiment sessions:"
      MENU_ITEMS=()
      for EXP in "${EXPERIMENTS[@]}"; do
        if session_exists "$EXP"; then
          MENU_ITEMS+=("$EXP")
        else
          MENU_ITEMS+=("$EXP [not running]")
        fi
      done
      MENU_ITEMS+=("Quit")

      PS3="Select a session to attach (or 'q' to quit): "
      select CHOICE in "${MENU_ITEMS[@]}"; do
        if [[ "$REPLY" =~ ^[qQ]$ ]]; then
          break 2
        elif [[ "$REPLY" -eq $((${#MENU_ITEMS[@]})) ]]; then
          break 2
        elif [[ -n "$CHOICE" ]]; then
          SESSION_NAME="${CHOICE%% *}"
          if session_exists "$SESSION_NAME"; then
            tmux attach-session -t "$SESSION_NAME"
            echo "Session '$SESSION_NAME' detached. Returning to menu."
          else
            echo "Session '$SESSION_NAME' is not running."
          fi
          break
        else
          echo "Invalid choice. Try again."
        fi
      done
    done
    ;;

  kill)
    for EXP in "${EXPERIMENTS[@]}"; do
      tmux kill-session -t "$EXP" 2>/dev/null && \
        echo "Killed session '$EXP'" || \
        echo "Session '$EXP' not running"
    done
    ;;

  *)
    echo "Unknown action: $ACTION"
    echo "Usage: $0 [train|list|kill]"
    exit 1
    ;;
esac
