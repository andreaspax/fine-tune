
# download and install uv
curl -fsSL https://get.uv.dev | sh

# Get current directory path and make available to subsequent commands
export DIR=$(pwd)

# Restart shell to ensure uv is in PATH
exec $SHELL

# go to the directory
cd $DIR

# install dependencies
uv sync

# activate the environment
source .venv/bin/activate

# config GitHub
git config --global user.name "Andreas P"
git config --global user.email "49064513+andreaspax@users.noreply.github.com"