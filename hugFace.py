from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj="app.py",                        
    path_in_repo="https://huggingface.co/spaces/JGonz/DogClassifier",                           
    repo_id="JGonz/DogClassifier",         
    repo_type="space",                               
    token="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"
)

print("File uploaded!")
