debugging = False
verbose = True
debugger_username = 'j35'
debugger_folder = ['/Users/j35/HFIR/CG1D/',
                   '/Volumes/JeanHardDrive/HFIR/CG1D/']
debugger_instrument_folder = {'CG1D': ["/Users/j35/HFIR/CG1D",
                                       "/Volumes/JeanHardDrive/HFIR/CG1D/",
                                       ],
                              'SNAP': ["/Users/j35/SNS/SNAP"],
                              'VENUS': ["/Users/j35/SNS/VENUS"],
                              }
# debugger_instrument_folder = {'CG1D': "/Volumes/JeanHardDrive/HFIR/CG1D",
#                               'SNAP': "/Volumes/JeanHardDrive/SNS/SNAP",
#                               'VENUS': "/Volumes/JeanHardDrive/SNS/VENUS"}
analysis_machine = 'bl10-analysis1.sns.gov'
project_folder = 'IPTS-24863-test-imars3d-notebook'
percentage_of_images_to_use_for_roi_selection = 0.05
minimum_number_of_images_to_use_for_roi_selection = 10
DEFAULT_CROP_ROI = [0, 510, 103, 404]
DEFAULT_BACKROUND_ROI = [5, 300, 5, 600]
DEFAULT_TILT_SLICES_SELECTION = [103, 602]
STEP_SIZE = 50  # for working with bucket of data at a time
