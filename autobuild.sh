#!/bin/bash

# Function to check if a command exists and install if missing
check_and_install() {
    if ! command -v $1 &> /dev/null; then
        echo "$1 not found. Installing $1..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y $1
        elif command -v yum &> /dev/null; then
            sudo yum install -y $1
        else
            echo "Error: Package manager not found. Please install $1 manually."
            exit 1
        fi
    else
        echo "$1 is already installed."
    fi
}

# Check for required commands
check_and_install wget
check_and_install curl
check_and_install git

# Step 1: Download and install Anaconda if not installed
if command -v conda &> /dev/null; then
    echo "Anaconda already installed."
else
    echo "Anaconda not found. Fetching the latest Anaconda installer URL..."
    
    # Automatically fetch the latest Anaconda installer URL
    NEWEST_NAME=$(curl -s https://repo.anaconda.com/archive/ | grep -oP 'Anaconda3-[0-9]{4}\.[0-9]{2}-[0-9]{1}-Linux-x86_64\.sh' | head -n 1)
    
    # Check if the URL was fetched successfully
    if [ -z "$NEWEST_NAME" ]; then
        echo "Error: Could not fetch the latest Anaconda download URL."
        exit 1
    fi

    DOWNLOAD_URL="https://repo.anaconda.com/archive/$NEWEST_NAME"
    echo "Downloading Anaconda from $DOWNLOAD_URL..."
    wget $DOWNLOAD_URL -O ~/anaconda.sh
    if [ $? -ne 0 ]; then
        echo "Error: Failed to download Anaconda."
        exit 1
    fi
    
    echo "Installing Anaconda..."
    bash ~/anaconda.sh -b -p $HOME/anaconda
    if [ $? -ne 0 ]; then
        echo "Error: Anaconda installation failed."
        exit 1
    fi

    echo 'export PATH="$HOME/anaconda/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    
    eval "$($HOME/anaconda/bin/conda shell.bash hook)"
    rm ~/anaconda.sh
    echo "Anaconda installed successfully."
fi

# Step 2: Create virtual environment with Python 3.11.5
if conda env list | grep -q "^IMCFN "; then
    echo "Environment 'IMCFN' already exists. Activating existing environment."
    source activate IMCFN
else
    echo "Creating virtual environment 'IMCFN' with Python 3.11.5..."
    conda create -y -n IMCFN python=3.11.5
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment 'IMCFN'."
        exit 1
    fi
    source activate IMCFN
    if [ $? -ne 0 ]; then
        echo "Error: Failed to activate virtual environment 'IMCFN'."
        exit 1
    fi
    echo "Virtual environment 'IMCFN' created and activated successfully."
fi

# Step 3: Clone the repository
echo "Cloning the Git repository..."
REPO_DIR="IMCFN_with_Machine_Unlearning"
if [ -d "$REPO_DIR" ]; then
    read -p "The directory 'IMCFN' already exists. Do you want to delete it? (Y/N): " choice
    if [ "$choice" == "Y" ] || [ "$choice" == "y" ]; then
        rm -rf "$REPO_DIR"
        echo "'IMCFN' directory deleted."
        git clone https://github.com/cchunhuang/IMCFN_with_Machine_Unlearning.git
        if [ $? -ne 0 ]; then
            echo "Error: Failed to clone the Git repository."
            exit 1
        fi
        cd IMCFN_with_Machine_Unlearning || { echo "Error: Repository folder not found."; exit 1; }
        echo "Git repository cloned successfully."
    else
        echo "Keeping the existing 'IMCFN' directory."
        cd IMCFN_with_Machine_Unlearning || { echo "Error: Repository folder not found."; exit 1; }
    fi
fi

# Step 4: Install requirements
echo "Installing required packages..."
pip install -r requirements_cpu.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install required packages."
    exit 1
fi
echo "All required packages installed successfully."

echo "Setup completed successfully. You can now activate the environment using 'conda activate IMCFN'."
