#!/bin/bash
#!/bin/bash

# Default to 10 if no argument is given
TOP_N=${1:-10}

for i in {1..22}; do
  host="cuda$i"
  echo "===== $(date) | $host ====="
  ssh "$host" "ps -eo user,pid,ppid,cmd,%cpu,%mem --sort=-%mem | head -n $((TOP_N + 1))"
  echo ""
done

