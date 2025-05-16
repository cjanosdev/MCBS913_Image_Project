# Step 1 Activate the Conda Environment from `image` directory

## How to activate and use the Conda Environment
1) Once you are logged onto Ron **via terminal or PUTTY client** ***(you cannot do this with cyberduck)*** navigate to the following directory: `/home/share/groups/mcbs913-2025/image`
- The command to navigate to a directory or file is `cd` so in the terminal if you type: `cd /home/share/groups/mcbs913-2025/image`this will bring you to the correct directory.

2) Once in the image directory you can type the following command to activate the conda environment: `conda activate ./image_proj_env/` I named our conda environment "image_proj_env" so this is what you are "activating".

3) If you want to "exit" or "stop" using the conda environment simply type the following command: `conda deactivate`. It is considered good practice to deactivate a conda environment when you are done using it.


# Step 2 - Installing Necessary coding dependencies:

1) Navigate to `/home/share/groups/mcbs913-2025/image/submission/code/image_proj`

2) Once in the directory you can install the project dependencies by running the following command: `poetry install`

# Step 3 - Running the command line tool (cell morphology analyzer):

1) Once the dependencies are installed with poetry we can run the command line tool. In the same directory (`/home/share/groups/mcbs913-2025/image/submission/code/image_proj`) run the following:
`cell_morphology_analyzer`

2) This starts the application and will give you a menu of options to choose from:
Running the Cell Morphology Analyzer...

    🔷 DeepThought Terminal Interface
    1. 💬 Start new conversation
    2. 📖 View past conversation
    3. ⚙️ Start new conversation using Static RAG
    4. ⚙️ Start new conversation using Dynamic RAG
    5. 📝 Export last conversation
    6. Run Automated Evaluation
    7. ❌ Quit
    Choose an option [1/2/3/4/5/6/7]: 

Option 1 starts an out-of-the-box conversation with DeepThought LLM.
Option 2 allows you to view past conversations (if there are any available)
Option 3 starts a new conversation with DeepThought LLM with the addition of a static RAG component.
Option 4 starts a new conversation with DeepThought LLM with the addition of a dyanmic RAG component.
Option 5 allows you to export the last conversation (if there are any available)
Option 6 will run the 5 baseline question evaluation with our 4 preselected flourescent microscopy images. It will run with what we have determined to be the best combination of image preprocessing parameters after doing numerous runs and comparing the results.
Option 7 simply quits the command line application.



# Changing the Default config for preprocessing parameters and the 'n' size for cosine similarity search

- If you are interested in playing around with preprocessing parameters for Option 6 of our Cell Morphology Analzyer command line tool you will need to tweak the code.

1) In `/home/share/groups/mcbs913-2025/image/submission/code/image_proj/source/cell_morphology_analyzer.py` find the function called: `run_parameter_sweep` this is the funciton called for Option 6 from the main menu. In there you will find the following:

        `prompt_modes = ["dynamic_rag", "control"]`
        `preprocessing_methods = ["default", "none"]`
        `top_n_facts_options = [6]`

You can change these althought I would recommend leaving prompt_modes and preprocessing_methods untouched. If you want to see what happens by changing the number of facts to pull for the cosine similiarity search then you can tweak the 'top_n_facts_options' to the desired value.

The parameter sweep configuration in its final form is currently:
preprocessing_param_sweep = {
            "gaussian_blur": {
                "ksize": (5, 5),
                "sigmaX": 0,
            },
            "canny": {
                "threshold1": 50,
                "threshold2": 150,
            },
            "in_range": {
                "lowerb": (100, 50, 50),
                "upperb": (140, 255, 255),
                "lowerg": (40, 50, 20),
                "upperg": (80, 255, 255),
                "lowerred1": (0, 70, 50),
                "upperred1": (10, 255, 255),
                "lowerred2": (170, 70, 50),
                "upperred2": (180, 255, 255),
            },
            "find_contours": {
                "mode": cv2.RETR_EXTERNAL,
                "contour_method": cv2.CHAIN_APPROX_SIMPLE,
            },
            "draw_contours": {
                "contourIdx": -1,
                "color": (0, 255, 0),
                "thickness": 2,
            },
            "add_weighted": {
                "alpha": 0.6,
                "beta": 0.4,
                "gamma": 0.0,
            }
        }


You can tweak individual values to your liking.


When run you Option 6 you will get output that looks something like:
✅ Pipeline Completed.
📋 Summary of Scores:
 
🧪 Summary for Query: 'What are the fluorescent dyes being used in this image if any? Please specify which organelles are being stained and why?'
    Default Preprocessing
   ✅ Total Score: 13 / 20
   📈 Accuracy: 65.0% for dynamic_rag
   ✅ Total Score: 14 / 20
   📈 Accuracy: 70.0% for control
 
    No Preprocessing:
   ✅ Total Score: 12 / 20
   📈 Accuracy: 60.0% for dynamic_rag
   ✅ Total Score: 14 / 20
   📈 Accuracy: 70.0% for control
 
🧪 Summary for Query: 'Do the cells in these images seem healthy? If not, what is the process they could undergo and how would you know?'
 
Default Preprocessing
   ✅ Total Score: 13 / 20
   📈 Accuracy: 65.0% for dynamic_rag
   ✅ Total Score: 16 / 20
   📈 Accuracy: 80.0% for control
 
No Preprocessing:
   ✅ Total Score: 14 / 20
   📈 Accuracy: 70.0% for dynamic_rag
   ✅ Total Score: 13 / 20
   📈 Accuracy: 65.0% for control
 
🧪 Summary for Query: 'Do you notice any morphological changes present in these images? If so what types and to what ratio are you able to compare puncta to filamentous networks?'
 
Default Preprocessing
   ✅ Total Score: 15 / 20
   📈 Accuracy: 75.0% for dynamic_rag
   ✅ Total Score: 14 / 20
   📈 Accuracy: 70.0% for control
 
No Preprocessing:
   ✅ Total Score: 14 / 20
   📈 Accuracy: 70.0% for dynamic_rag
   ✅ Total Score: 15 / 20
   📈 Accuracy: 75.0% for control
 
🧪 Summary for Query: 'These images have two pathways, one showing the basics of staining to certain organelles while others show invasion. What is being invaded into these cells? Which organelles are these morphology changes occurring in these images?'
 
Default Preprocessing
   ✅ Total Score: 12 / 20
   📈 Accuracy: 60.0% for dynamic_rag
   ✅ Total Score: 14 / 20
   📈 Accuracy: 70.0% for control
 
No Preprocessing:
   ✅ Total Score: 13 / 20
   📈 Accuracy: 65.0% for dynamic_rag
   ✅ Total Score: 11 / 20
   📈 Accuracy: 55.0% for control
 
🧪 Summary for Query: 'How many of the cells in these images are undergoing cell death? Give me a number.'
 
Default Preprocessing
   ✅ Total Score: 11 / 20
   📈 Accuracy: 55.0% for dynamic_rag
   ✅ Total Score: 8 / 20
   📈 Accuracy: 40.0% for control
 
No Preprocessing:
   ✅ Total Score: 10 / 20
   📈 Accuracy: 50.0% for dynamic_rag
   ✅ Total Score: 8 / 20
   📈 Accuracy: 40.0% for control
 
📊 OVERALL FINAL SCORES BY PROMPT MODE:
   🧠 Mode: dynamic_rag
      ✅ Score: 127 / 200
      📈 Accuracy: 63.5%
   🧠 Mode: control
      ✅ Score: 127 / 200
      📈 Accuracy: 63.5%
 
🏁 CONTROL VS DYNAMIC RAG COMPARISON:
   🔍 Dynamic RAG Accuracy: 63.5%
   🧪 Control Accuracy:     63.5%
   🎯 CONTROL outperformed by 0.0 percentage points
 
📊 OVERALL COMBINED ACCURACY ACROSS ALL MODES:
   ✅ Total Score: 254 / 400
   📈 Accuracy: 63.5%
 
🧪 DEFAULT VS NONE PREPROCESSING COMPARISON:
   🧪 Default Accuracy: 65.0%
   🧪 None Accuracy:    62.0%
   ✅ DEFAULT outperformed NONE by 3.0 percentage points


This goes through each baseline question and compares dynamic rag to the control along with default preprocessing against no-preprocessing.