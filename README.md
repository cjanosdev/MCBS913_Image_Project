# MCBS913_Image_Project

# Instructions for access Ron:
1: Open the terminal on your computer (if using a Mac the default terminal) if on windows use PUTTY client.

2: Connect to Ron with: `ssh yourusernameatunh@ron.sr.unh.edu` or use PUTTY client.


# Instructions on how to use CyberDuck to connect to Ron and Download or Upload files,data,etc..
1) Open CyberDuck
2) Change the dropdown to SFTP (SSH File Transfer Protocol)
3) In the Server box put: `ron.sr.unh.edu`
4) In the Username box put your unh login username so for example mine is: `clj1019`
5) In the Password box put your unh password.
6) In SSH Private Key drop down make sure `None` is selected.
7) Click the connect button.
<img width="492" alt="Screenshot 2025-03-06 at 11 18 59 AM" src="https://github.com/user-attachments/assets/2265f794-348a-4bb6-9818-4f7e0a214f4b" />

## How to Downlaod or Upload files using CyberDuck
1) Once you are connected to Ron in CyberDuck you will see that you are at your `/home/users/whateveryourusernameis` path location.
2) You will likely be wanting to upload or download files to our shared folder location which is `/home/share/groups/mcbs913-2025/image`. Use the drop down to navigate to this directory.
3) Once in the image directory you can choose to upload data, images, files, etc. by simply dragging and dropping them into the desired folder in CyberDuck. You will see a small pop-up that indicates the file is being uploaded to Ron using CyberDuck.
4) Similiary to download something that is on Ron using CyberDuck right click the file you want and select `Download`. You will see a similar pop-up to that which is displayed for uploading files.
<img width="597" alt="Screenshot 2025-03-06 at 11 26 03 AM" src="https://github.com/user-attachments/assets/bf1db3f5-dd6d-4eb1-b9db-80b9ec3f6c5f" />



# How to activate and use the Conda Environment
1) Once you are logged onto Ron **via terminal or PUTTY client** ***(you cannot do this with cyberduck)*** navigate to the following directory: `/home/share/groups/mcbs913-2025/image`
- The command to navigate to a directory or file is `cd` so in the terminal if you type: `cd /home/share/groups/mcbs913-2025/image`this will bring you to the correct directory.

2) Once in the image directory you can type the following command to activate the conda environment: `conda activate image_proj_env` I named our conda environment "image_proj_env" so this is what you are "activating".

3) If you want to "exit" or "stop" using the conda environment simply type the following command: `conda deactivate`. It is considered good practice to deactivate a conda environment when you are done using it.

# Packages directly installed to our environment
As of 03/06/20225 we have the following installed to our environment:

| Package Name            | Version  | Date Added |
|-------------------------|----------|------------|
| pyton                   | 3.12     | 03/06/2025 |
| opencv                  | 4.10.0   | 03/06/202  |
| huggingface_hub         | 0.24.6   | 03/06/2025 |
| datasets                | 2.19.1   | 03/06/202  |
| safetensors             | 0.4.5    | 03/06/2025 |
| transformers            | 4.49.0   | 03/06/202  |
| pyton_abi               | 3.12     | 03/06/2025 |
| tokenizers              | 0.21.0   | 03/06/202  |
| pyimagej                | 1.6.0    | 03/06/202  |
| openjdk                 | 11.0.1   | 03/06/202  |
| lftp                    | 11.0.1   | 03/06/202  |


# Downloading Data from ICR using LFTP
1) In a terminal window login into Ron
2) Once connected to Ron and in the `/home/share/groups/mcbs913-2025/image` directory. First activate our conda environment:
   >> `conda activate image_proj_env`
4) Once the Conda environment is active will connect to the IDR using LFTP (Command-Line FTP Client via Conda):
   >> `lftp ftp.ebi.ac.uk`

5) Navigate to the IDR databse directory:
>> `[lftp ftp.ebi.ac.uk:~> cd /pub/databases/IDR/
cd ok, cwd=/pub/databases/IDR   `

4) Use the `ls` command to list all studies in the IDR:
   > But we want idr0080-way-perturbation in particular so we will do:
   >>   `[lftp ftp.ebi.ac.uk:/pub/databases/IDR> ls idr0080-*`
  > This just filters down the list to studies that have "idr0080-" in the name.
5) We can use the `get` command to download a specific file.
> Note you can only use `get` for a single file. In order to download a whole directory (and its content) we need to do whats called a **recursive** download using the `mirror` command note I have `/path/to/local/directory` in the below command change this to `.` if you just want it to download to the current directory you are in or specify the desired path:
>> `mirror -c idr0080-way-perturbation /path/to/local/directory`

<img width="900" alt="Screenshot 2025-03-06 at 12 30 11 PM" src="https://github.com/user-attachments/assets/03bc3464-3680-4316-8899-47b30884b9dc" />


6) When you are done dowloading a file type `bye`to close the connection to the IDR.
