import boto3
import requests
import time
import json
# Replace 'YOUR_ACCESS_KEY' and 'YOUR_SECRET_KEY' with your actual AWS access key and secret key
import os
# Replace 'YOUR_ACCESS_KEY' and 'YOUR_SECRET_KEY' with your actual AWS access key and secret key
access_key = os.getenv('AWS_ACCESS_KEY')
secret_key = os.getenv('AWS_SECRET_KEY')

# Replace 'us-west-2' with your desired AWS region
region = 'us-east-1'

# Configure the AWS credentials and region
boto3.setup_default_session(aws_access_key_id=access_key, aws_secret_access_key=secret_key, region_name=region)
def transcribe_audio_from_url(url,language_code):
    """
    Transcribes audio from a given URL using Amazon Transcribe.

    Args:
        url (str): The URL of the MP3 file to transcribe.

    Returns:
        None
    """
    # Replace 'bucket_name' with the name of your S3 bucket
    bucket_name = 'llmaudiofiles'
    

    # Replace 'file_path' with the path to your local audio file
    # file_path = '/path/to/your/audio/file.mp3'
    
    
    
    file_name = url.split('/')[-1]

    # Create the file path
    file_path = './downloads/' + file_name

    # Create an S3 client
    s3_client = boto3.client('s3')
    
    # Generate a unique file name for the uploaded file
    uploaded_file_name = file_name

    # Upload the audio file to S3 with the same file name as in the URL
    s3_client.upload_file(file_path, bucket_name, uploaded_file_name)

    # Upload the audio file to S3
    # s3_client.upload_file(file_path, bucket_name, 'audio.mp3')
    
    # Get the URI of the uploaded file
    file_uri = f"s3://{bucket_name}/{uploaded_file_name}"

    # Get the URL of the uploaded file
    # url = f"https://{bucket_name}.s3.amazonaws.com/audio.mp3"

    # Call the transcribe_audio_from_url function with the URL
    
    
    
    
    # Create a Transcribe client
    transcribe_client = boto3.client('transcribe')

    # Specify the transcription job parameters
    # job_name = 'transcription-job'  # Specify a unique job name
    job_name = "transcription-job-" + str(int(time.time()))
    media = {'MediaFileUri': file_uri}  # Specify the URL of the MP3 file
    language_code = language_code  # Specify the language code

    # Start the transcription job
    response = transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        Media=media,
        MediaFormat='mp3',
        LanguageCode=language_code
    )
    
    # Wait for the transcription job to complete
    while True:
        response = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        status = response['TranscriptionJob']['TranscriptionJobStatus']
        if status in ['COMPLETED', 'FAILED']:
            break
        time.sleep(5)

    # Get the transcription job result
    if status == 'COMPLETED':
        transcription_url = response['TranscriptionJob']['Transcript']['TranscriptFileUri']
        # Download the transcription file
        transcription_file = requests.get(transcription_url)
        # Save the transcription to a file
        
        # Convert the transcription content to JSON
        transcription_json = json.loads(transcription_file.content)
        
        return transcription_json

        # Save the transcription as JSON
        with open('transcription.json', 'w') as file:
            json.dump(transcription_json, file)

        print("Transcription saved to transcription.json")
        
        with open('transcription.json', 'wb') as file:
            file.write(transcription_file.content)
            print("Transcription saved to transcription.json")
    else:
        print("Transcription job failed")

    # Print the response
    return(response)

# Replace 'url_of_mp3' with the actual URL of the MP3 file
# transcribe_audio_from_url('http://3.84.5.107/static/1.mp3')