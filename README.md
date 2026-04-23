Keep strict folder structure if directly want to run the code.
cropharvest_datas
|
|___ features
|    |
|    |___ arrays
|    |
|    |
|    |
|    |
|    |___ normalized_dict.h5
|
|____ labels.geojson
|
|____ data
|
|
|    |____ boundry
|    |
|    |____ final_output
|    |
|    |____ history
|    |
|    |____ live
|    |
|    |____live_weather
|    |
|    |____ state
|    |
|    |____ wheat_mask
|
|____ all_integrated.py
|
|____ app.py
|
|____ wheat_classification.py



Run wheat_classification.py first

then all_intgerated.py

then in your powershell  streamlit run app.py

Note:- this project relies on google earth engine. you will have create profile and authenticate change project name to run it.
