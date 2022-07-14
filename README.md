# Fire Segmentation neural network
NN for wildfire detection.

To execute this code, download datasets 9 and 10 from the [FLAME dataset](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs) and extract them to the appropriate folder.



## Environment configuration

This project uses Julia 1.7. Install the Julia executable on your system and check that the command `julia` actually works in the terminal. If it is, open a terminal on this project's repository and initiate Julia. Open the package terminal by typing `]`. The REPL dialog should turn blue and display the Julia version you are using. Then type `activate .` and `[ENTER]`. This should activate the repositoy environment, displaying the name of the current folder between the parenthesis that earlier contained the Julia version. Finally give the command `instantiate` to install all necessary packages. You can test the installation by leaving the package terminal (with a simple backspace) and trying `using PACKAGE` for every package you wished to install. Please note that this procedure reads from the files `Manifest.toml` and `Project.toml`, which describe the dependencies. Do not delete or modify them manually, and do not forget to commit them if any dependency changes have taken place. If you are using VS Code, do not forget to set the executable path to the correct location in the Julia extension's settings