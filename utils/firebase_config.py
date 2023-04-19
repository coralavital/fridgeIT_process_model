from firebase_admin import credentials, initialize_app, db, firestore, storage


cred = credentials.Certificate("./utils/FBserviceAccountKey.json")

initialize_app(cred, {'storageBucket': 'fridgeit-d17ae.appspot.com'})

db = firestore.client()
bucket = storage.bucket()

user_document = db.collection(
        'EPaDIxTXxINXm88w7xoBPRaNcFh1').document('user_data')


cropped_folder_storage_path = 'cropped/'
history_cropped_storage_path = 'cropped_history/'