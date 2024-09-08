#!/bin/bash

# Path to the requirements.txt file
REQUIREMENTS_FILE="req.txt"

# Check if the requirements file exists
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "requirements.txt file not found!"
    exit 1
fi

# Loop through each line in the requirements.txt file
while IFS= read -r package || [ -n "$package" ]; do
    # Install the package using pip
    echo "Installing $package"
    pip install "$package"

    # Check if the installation was successful
    if [ $? -eq 0 ]; then
        echo "$package installed successfully"
    else
        echo "Failed to install $package"
        exit 1
    fi
done < "$REQUIREMENTS_FILE"

echo "All packages installed successfully."
