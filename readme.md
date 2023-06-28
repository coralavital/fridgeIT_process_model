# INSTALLATION
To get started with the FridgeIT pipelines from Hugging Face, please follow the steps below:
1. Create the directory where you want to clone the FridgeIT pipelines repository.
2. Open the terminal and navigate to the directory you just created or open the terminal from this directory.
3. Run the following command to install Git Large File Storage (LFS) for handling large files: <br>
    ```
   git lfs install
   ```
4. Clone the Hugging Face repository by running the following command:<br>
    ```
   git clone https://huggingface.co/coralavital/fridgeIT_pipeline.
    ```
5. Open your workspace from the `fridgeIT_pipeline` directory.
6. In the workspace, open the terminal, and execute the following commands:
    - Create a new Python virtual environment:<br>
      ```
      python3.9 -m venv venv
      ```
    - Activate the virtual environment:
      On macOS/Linux:<br>
        ```
      source venv/bin/activate
        ```
      On Windows:<br>
        ```
        venv\Scripts\activate.bat
      ```
    - Install necessary dependencies:<br>
      ```
      pip3 install -r requirements.txt
      ```
7. After completing the installation steps, you can run the FridgeIT pipelines from the Python file named "process.py" inside the "pipelines" directory in your workspace.
