from fastapi import FastAPI,HTTPException
from tinydb import TinyDB, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests


from typing import ClassVar
from pydantic import BaseModel
from pdfkit import from_string
from schema import BlogCoverImg, WriterPointOfView, BlogGenerationMode, BlogLanguage, MediaLanguage, BlogTone, BlogSize, BlogGenerationMode
from pytube import YouTube
from pytube.exceptions import RegexMatchError
from youtube_transcript_api import YouTubeTranscriptApi
import datetime

from services.transcription_service import transcribe_audio_from_url
from fastapi.staticfiles import StaticFiles

from schema import Language , AWSTranscriptCode, Model
from pytube import YouTube


# from get_transcript import generate_questions


import openai
import json
import time
import tiktoken 

import time 

import logging

logger = logging.getLogger(__name__)  # Use the current module's name
logger.setLevel(logging.DEBUG)  # Capture DEBUG level logs and above 

# Create a file handler to store logs
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.DEBUG)

# Create a formatter to specify the log message format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)



import re
# import logging
def extract_json(text_response):
    # This pattern matches a string that starts with '{' and ends with '}'
    pattern = r'\{[^{}]*\}'

    matches = re.finditer(pattern, text_response)
    json_objects = []

    for match in matches:
        json_str = match.group(0)
        try:
            # Validate if the extracted string is valid JSON
            json_obj = json.loads(json_str)
            json_objects.append(json_obj)
        except json.JSONDecodeError:
            # Extend the search for nested structures
            extended_json_str = extend_search(text_response, match.span())
            try:
                json_obj = json.loads(extended_json_str)
                json_objects.append(json_obj)
            except json.JSONDecodeError:
                # Handle cases where the extraction is not valid JSON
                continue

    if json_objects:
        return json_objects
    else:
        return None  # Or handle this case as you prefer

def extend_search(text, span):
    # Extend the search to try to capture nested structures
    start, end = span
    nest_count = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            nest_count += 1
        elif text[i] == '}':
            nest_count -= 1
            if nest_count == 0:
                return text[start:i+1]
    return text[start:end]


def get_maxtoken(text,model,max_token):
    try:
        encoding = tiktoken.encoding_for_model(model)
        print('Successfully obtained max token')
        return max_token-1-len(encoding.encode(text))
    except Exception as e:
        print('Error while obtaining max token: ')
        raise

def get_token(text,model):
    try:
        encoding = tiktoken.encoding_for_model(model)
        print('Successfully obtained token')
        return len(encoding.encode(text))
    except Exception as e:
        print('Error while obtaining token: ')
        raise

def get_message(text):
    try:
        content="""
        Using the following job description, please generate  multiple-choice questions that test the candidate's suitability for the role, it's not for testing their understanding of the key requirements and responsibilities. Each answer is scored from 1-5 with the best answer being 5, 1 being the worst answer. output JSON  the following structure: [{question , answers: [{answer, score, rationale}]}]:

         {text}
        """.replace("{text}",text)
        
        content =text
        
        message= [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": content
            }
        ]
        print('Message created successfully.')
        return message
    except Exception as e:
        print('Error while creating message: ')
        raise


def generate_questions(text):
    start = time.time()

    try:
        questions = []
        messages=get_message(text)
        for _ in range(0,3):
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4",  # Replace with the correct model
                        messages=messages,
                        max_tokens=4000,
                    )

                    # Assuming the OpenAI API returns the questions in the desired format
                    generated_questions = json.loads(response['choices'][0]['message']['content'].strip())
                    questions.extend(generated_questions)

                    # Add the last assistant's message to the conversation to build on the previous response
                    messages.append({
                        "role": "assistant",
                        "content": response['choices'][0]['message']['content']
                    })
                    if len(questions) >=8:
                        break
                except Exception as e:
                    print('Error while generating questions: ')

        end = time.time()
        print(f"Time taken: {end - start} seconds")
        return questions
    except Exception as e:
        print('Error in generate_questions function: ')
        raise



import google.generativeai as genai
# genai.configure(api_key="AIzaSyBfV310G0cHkZyzUXuzN5pejEVg9i7fNIw")



prompt = """Welcome, Video Summarizer! Your task is to distill the essence of a given YouTube video transcript into a concise summary. Your summary should capture the key points and essential information, presented in bullet points, within a 250-word limit. Let's dive into the provided transcript and extract the vital details for our audience."""

def generate_gemini_content(transcript_text, prompt):
    model = genai.GenerativeModel("gemini-pro")
    # model = genai.GenerativeModel("gemini-1.5-pro-latest")

    response = model.generate_content(prompt + transcript_text)
    return response.text




# class RequestBody(BaseModel):
#     lang: ClassVar[str]

app = FastAPI()

app.mount("/static", StaticFiles(directory="downloads"), name="downloads")


# Create a TinyDB database instance
db = TinyDB("my_database.json")

videos_table = db.table('videos')
transcript_table = db.table('transcript')
summarization_table = db.table('summarization')
articles_table = db.table('articles')
status_table = db.table('status')
settings_table = db.table('settings')


settings = settings_table.all()
settings[0]["geminiKey"] if settings else {"message": "No settings found"}
logger.info(f"Settings: {settings} gemini key: {settings[0]['geminiKey']}")

# AIzaSyAIFlCjGEr-0eGdcH3YT8kCCeNCWfZspn0


try:
    genai.configure(api_key="AIzaSyAIFlCjGEr-0eGdcH3YT8kCCeNCWfZspn0")
except Exception as e:
    logger.error(f"Error configuring genai: {e}")




# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sample endpoint
@app.get("/items")
def read_items():
    return db.all() 



# To run the app: uvicorn main:app --reload 

# class RequestBody(BaseModel):
#     youtube_url: str
#     model: str
#     lang=str


@app.get("/videos")
def get_videos():
    return videos_table.all()


    
@app.post("/add_task")
def add_task(url: str, lang: str):
    # Perform the necessary operations with the url
    # For example, you can make a request to the url and process the response
    # response = requests.get(url)
    
    audio_url="https://www.youtube.com/watch?v=Zi_XLOBDo_Y"
    
    # Check if the request was successful
    video_id = None
    if "youtube.com" in url:
        video_id = url.split("v=")[1]
    
    if video_id:
        # Check if the video_id already exists in the database
        if db.contains(Query().video_id == video_id):
            return {"message": "Video ID already exists"}
        
        # Add the task to the database or perform any other necessary operations
        videos_table.insert({"url": url, "lang": lang, "video_id": video_id, "audio_url": audio_url})
        return {"message": "Task added successfully"}
    else:
        return {"message": "Failed to add task"}
    
    
    
@app.get("/get_transcript",tags=["transcript"])
def get_transcript(video_id: str):
    # Retrieve the transcript for the given video_id
    transcript = transcript_table.search(Query().video_id == video_id)
    
    # Check if the transcript was found 
    if transcript:
        return {"transcript": transcript}
    else:
        return {"message": "Transcript not found"}



@app.post("/get_transcript",tags=["transcript"])
def get_transcript(video_id: str,servicename: str):
    # Retrieve the transcript for the given video_id
    transcript = "retrieve_transcript(video_id)"
    
    # Check if the transcript was found 
    
    if servicename == "aws_transcript":
        transcript_table.insert({"video_id": video_id, "service":servicename,"transcript": transcript})
        return {"transcript": transcript}

    
    

@app.get("/get_summarization",tags=["summarization"])
def get_summarization(video_id: str):
    # Retrieve the transcript for the given video_id
    summarization = summarization_table.search(Query().video_id == video_id)
    
    # Check if the transcript was found 
    if summarization:
        return {"summarization": summarization}
    else:
        return {"message": "Summarization not found"}
    
@app.post("/get_summarization",tags=["summarization"])
def get_summarization(video_id: str, transcript: str):
    # Perform the summarization
    summarization = "perform_summarization(transcript)"
    
    # Check if the summarization was successful
    if summarization:
        summarization_table.insert({"video_id": video_id, "summarization": summarization})
        return {"summarization": summarization}
    else:
        return {"message": "Failed to summarize the transcript"}
    
    

# @app.get("")
    
@app.get("/get_article",tags=["articles"])
def get_articles(video_id: str, article_id: str):
    # Retrieve the articles for the given video_id and article_id
    articles = articles_table.search((Query().article_id == article_id))
    
    # Check if the article was found 
    if articles:
        article = articles[0]
        title = article.get("title")
        transcript = article.get("transcript")
        summarization = article.get("summarization")
        article_body= article.get("article")
        
      
        
        
        keywords = article.get("keywords")
        
        if "," in keywords:
            keywords = keywords.split(",")
        
        
        if "،" in keywords:
            keywords = keywords.split("،")
        
        return {
            "article_id": article_id,
            "title": title,
            "transcript": transcript,
            "summarization": summarization,
            "article": article_body,
            "keywords": keywords
        }
    else:
        return {"message": "Article not found"}
    
    
    # Check if the articles were found 
    if article:
        return {"article_id":article_id,"title":title,"article": article, "video_id": video_id,"transcript":transcript, "summarization":summarization, "keywords":keywords}
    else:
        return {"message": "Articles not found"}
    
    
@app.get("/get_all_articles", tags=["articles"])
def get_all_articles():
    # Clear the articles table
    # articles_table.truncate()
    
    # Retrieve all articles from the articles table
    all_articles = articles_table.all()
    
    # Check if any articles were found
    if all_articles:
        return all_articles
    else:
        return {"message": "No articles found"}

@app.post("/get_articles",tags=["articles"])
def get_articles(video_id: str, transript: str):
    # Perform the article generation
    articles = "generate_articles(summarization)"
    
    # Check if the article generation was successful
    if articles:
        articles_table.insert({"video_id": video_id, "articles": articles})
        return {"articles": articles}
    else:
        return {"message": "Failed to generate articles"}

    
@app.get("/download_article",tags=["articles"])

def download_article(video_id: str, article_id: str,type: str):
    # Retrieve the article for the given video_id
    article = articles_table.search(Query().video_id == video_id)
    
    title="This is an example title"
    body="<h1>This is an example article for the video.</h1>"
    
    if type=="pdf":
        # Convert HTML body to PDF
        pdf = from_string(body, False)

        # Save PDF to file
        pdf_path = f"./static/{title}.pdf"
        with open(pdf_path, "wb") as file:
            file.write(pdf)
            
        return {"pdf_path": pdf_path}
    if type == "docx":
        # Convert HTML body to DOCX
        docx = from_string(body, False)

        # Save DOCX to file
        docx_path = f"./static/{title}.docx"
        with open(docx_path, "wb") as file:
            file.write(docx)
            
        return {"docx_path": docx_path}
        
    
    # Check if the article was found 
    if article:
        return {"title":title,"article": article}
    else:
        return {"message": "Article not found"}
     

@app.post("/save_article", tags=["articles"])
async def save_article(request_body: dict):
    # Extract the necessary fields from the request body
    video_id = request_body.get("video_id")
    article_id = request_body.get("article_id")
    title = request_body.get("title")
    article = request_body.get("article")
    
    # Save the article to the database or perform any other necessary operations
    # articles_table.update({"article_id": article_id}, {"video_id": video_id, "article_id": article_id, "title": title, "article": article})
    
    articles_table.update({"video_id": video_id, "article_id": article_id, "title": title, "article": article}, Query().article_id == article_id)
    return {"status":"success","message": "Article saved successfully"}
    # Save the article to the database or perform any other necessary operations
    articles_table.insert({"video_id": video_id, "article_id": article_id, "title": title, "article": article})
    return {"message": "Article saved successfully"}

    # If the article saving was not successful
    return {"message": "Failed to save the article"}, 400


    
class CreateArticelRequestBody(BaseModel):
    blog_generation_mode: str = "auto-Pilot"
    blog_language: str = "english"
    blog_size: str = "Medium"
    blog_tone: str = "engaging"
    media_language: str = "english"
    writer_point_of_view: str = "third-person"
    youtube_url: str
    article_id:str
    




@app.post("/create_article", tags=["articles"])
async def create_article(request_body: CreateArticelRequestBody):
    writer_point_of_view = request_body.writer_point_of_view
    blog_generation_mode = request_body.blog_generation_mode
    blog_language = request_body.blog_language
    
    media_language = request_body.media_language
    
    
    blog_tone = request_body.blog_tone
    blog_size = request_body.blog_size
    youtube_url = request_body.youtube_url
    article_id=request_body.article_id
    
    # Create a logger
    # logger = logging.getLogger(__name__)

    # Log the article ID
    logger.info("#########article_id##########")
    
    try:
        # Attempt to create a YouTube object with the provided URL
        youtube_obj = YouTube(youtube_url)
        
        video_id = youtube_obj.video_id
        
        # Check if the video duration is less than 10 minutes
        if youtube_obj.length > 600:
            # Video duration is less than 10 minutes
            logger.error("Video duration is more than 10 minutes")

            return {"status": "failed", "message": "Video duration is more than 10 minutes"}
        
        else:
            
            
            article_id
            # try:
            # article_id = f"{video_id}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            # article_id=article_id
            def update_status(article_id: str, video_id: str, status: str, step: str, percentage: int):
                if not status_table.search(Query().article_id == article_id):
                    status_table.insert({"video_id":video_id,"article_id": article_id, "video_id": video_id, "status": status, "step": step, "percentage": percentage})
                else:
                    status_table.update({"status": status, "step": step, "percentage": percentage}, Query().article_id == article_id)
                logger.info(f"Updated status for article_id: {article_id}, video_id: {video_id}, status: {status}, step: {step}, percentage: {percentage}")


            update_status(article_id, video_id, "processing", "start transcribing", 0)
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[media_language])
            except Exception as e:
                update_status(article_id, video_id, "failed", "failed get transcript for this video", 0)

                logger.error(f"Failed to get transcript: {str(e)}")
                return {"status": "failed", "message": "Failed to get transcript"}
            
            transcript_text = ""
            for segment in transcript:
                if segment["text"].strip() != "":
                    transcript_text += segment["text"] + " "
                    

                    
            # Remove trailing whitespace
            transcript_text = transcript_text.strip()
            # Generate a unique article ID by combining video ID with timestamp
            sentences = transcript_text.split(".")

            
            
            # removed article id
            # article_id = f"{video_id}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            update_status(article_id, video_id, "processing", "end transcribing", 20)
            
            update_status(article_id, video_id, "processing", "start summarization", 40)

            summurization = "summurize_transcript(transcript)"
            settings = settings_table.all()
            prompt=settings[0]["transcriptSummarization"]
            
            # return {"status": "failed", "message": prompt}
            
            # prompt = settings_table.get(Query().key == "transcriptSummarization")
            if prompt is None or len(prompt.split()) < 3:
                return {"status": "failed", "message": "Please configure prompt in settings"}
            # prompt = prompt["value"]
            
            # return {"status": "failed", "message": prompt}
            
            prompt=prompt.replace("{{{writer_point_of_view}}}",writer_point_of_view)
            
            prompt=prompt.replace("{{{blog_generation_mode}}}",blog_generation_mode)
            prompt=prompt.replace("{{{blog_language}}}",blog_language)
            prompt=prompt.replace("{{{media_language}}}",media_language)
            prompt=prompt.replace("{{{blog_tone}}}",blog_tone)
            prompt=prompt.replace("{{{blog_size}}}",blog_size)
            
            prompt=prompt.replace("{{{transcript}}}",transcript_text)
            
            
            # logger.info(f"Prompt: {prompt}")
            
            
            
            
            # summary=generate_questions(prompt)
            
            # return summary
            try:
                summary = generate_gemini_content(transcript_text, prompt)
                logger.info(f"Gemini Response : {summary}")

            except Exception as e:
                logger.error(f"Failed to generate summary: {str(e)}")
                # return {"status": "failed", "message": "Failed to generate summary"}
            
            
            
            # return  {"status": "failed", "message": summary}
            def strip_control_characters(text):
                new_text = ""
                for char in text:
                    if 0 <= ord(char) <= 31:  # Check if it's a control character 
                        continue                # Skip it
                    new_text += char
                return new_text
            
            # # retu
            
            # summary = strip_control_characters(summary)
            
            # import json
            # json_body = json.loads(summary, strict=True)
            
        
            
            try:
                json_body=summary.replace("```json\n","")
                json_body=json_body.replace("\n```","")
                json_body = strip_control_characters(json_body)
                json_body = json.loads(json_body, strict=True)
                
                # json_body = extract_json(json_body)
            except Exception as e:
                
                json_body = str(e)
                
            logger.info(f"JSON Body Response : {json_body}")

                
                

            
            # return  {"status": "failed", "message": json_body , "summm":summary}
            
            
            update_status(article_id, video_id, "processing", "end summarization", 60)
            
            update_status(article_id, video_id, "processing", "start article generation", 80)
            
            article = "generate_article(summurization)"
            title="This is an example title"
            keywords=["keyword1","keyword2","keyword3"]
            
            update_status(article_id, video_id, "processing", "end article generation", 90)
            
            # article_prompt="make this article clean and pretty in HTML format"
            # article_pretty = generate_gemini_content(json_body["content"], article_prompt)
            
            article_pretty = json_body["content"]
            
            
            if video_id.endswith("?"):
                video_id = video_id[:-1]
            if video_id[-1] == "?":
                video_id = video_id[:-1]
            img=f"http://img.youtube.com/vi/{video_id}/0.jpg"
            
            img = f'<img src="{img}" alt="Cover Image" style="width:100%;object-fit:cover; margin-top: 4px; margin-bottom: 4px;">'
            
            article_pretty = img + json_body["content"]

            # Save the article to the database or perform any other necessary operations
            articles_table.insert({"video_id": video_id, "article_id": article_id, "title": json_body["title"], "article": article_pretty, "transcript": transcript_text, "summarization": json_body["summarization"], "keywords": json_body["keywords"]})
            logger.info("Inserted successfully ============= : video_id={}, article_id={}, title={}, article={}, transcript={}, summarization={}, keywords={}".format(video_id, article_id, json_body["title"], article_pretty, transcript_text, json_body["summarization"], json_body["keywords"]))

            update_status(article_id, video_id, "success", "end article saving", 100)
            
            return {"status": "success", "message": json_body}
            # return {"status": "failed", "message": json_body}

                
                
       
                
            # except Exception as e:
            #     return {"status": "failed", "message": str(e)}
            return {"status": "failed", "message": transcript_text}

            # Video duration is greater than or equal to 10 minutes
            return {"status":"success","message": "Video duration is valid", "video_id": video_id,"article_id": "1"}

            # return {"status": "success", "message": "Video duration is valid"}
        
        # If the object is created successfully, it means the URL is valid
        # return {"status":"success","message": "Valid YouTube URL", "video_id": video_id}
    except RegexMatchError:
        # If a RegexMatchError is raised, it means the URL is not a valid YouTube URL
        
        # raise HTTPException(status_code=400, detail="Invalid YouTube URL111")

        return {"status":"failed","message": "Invalid YouTube URL"}

    
    # Perform the necessary operations to create the article
    # ...
    # Return the created article
    return {"message": "Article created successfully"}
    
    

@app.get("/article_status", tags=["articles"])
def get_status(article_id: str):
    # Check the status of the article with the given article_id
    # You can check the status in the database or any other storage mechanism
    
    # Assuming you have a status field in the articles table
    # status = status_table.get(Query().article_id == article_id).get("status")
    
    # video_id = articles_table.get(Query().article_id == article_id).get("video_id", "")

    
    
    status = status_table.get(Query().article_id == article_id)
    # status_table.update({"video_id": video_id}, Query().article_id == article_id)
    return status
    status = "processing"
    
    
    if status == "processing":
        return {"status": "processing", "step": "transcribing","percentage":"20"}
    elif status == "success":
        return {"status": "success"}
    elif status == "failed":
        return {"status": "failed"}
    else:
        return {"message": "Invalid article_id"}


@app.post("/save_settings", tags=["settings"], response_model=dict)
async def save_settings(request_body: dict):
    # Save the settings to the database or perform any other necessary operations
    # ...
    # return request_body
    settings_table.update(request_body, doc_ids=[1])
    return {"status": "success", "message": "Settings saved successfully"}

@app.get("/get_settings", tags=["settings"], response_model=dict)
async def get_settings():
    # Retrieve the settings from the database or any other storage mechanism
    # ...
    settings = settings_table.all()
    return settings[0] if settings else {"message": "No settings found"}
    return settings[0] if settings else {"message": "No settings found"}


@app.delete("/delete_article/{blogId}", tags=["articles"])
def delete_article(blogId: str):
    
    # Perform the necessary operations to delete the article with the given blogId
    # ...
    status = "success"  # or "failed"
    try:
        articles_table.remove(Query().article_id == blogId)
    except Exception as e:
        status = "failed"
        return {"status": status, "message": f"Failed to delete article with blogId {blogId}: {str(e)}"}
    return {"status": status, "message": f"Article with blogId {blogId} deleted successfully"}

from services.youtube_service import YouTubeDownloader


downloader = YouTubeDownloader()


@app.get("/download-audio",tags=["audio_aws"])
async def download_audio(url: str):
    

    try:

        audio_file_path = downloader.download_youtube_audio(url=url)
        
        
        
        base_url = "http://api.findapply.com/static" 
        audio_file_path = audio_file_path.split("/downloads")[1]
        audio_file_path = base_url + audio_file_path

   

        return {"mp3":audio_file_path,"status":"succefull"} 
    except HTTPException as e:
        return {"status":"failed","message":e.detail}
        raise HTTPException(status_code=e.status_code, detail=e.detail)

@app.post("/transcribe-audio", tags=["audio_aws"])

async def transcribe_audio(audio_url: str, lang: AWSTranscriptCode):
    
    # print("audio_url", audio_url)
    
        
    try:
       
        transcription_result = transcribe_audio_from_url(audio_url, language_code=lang)

        return transcription_result
        # return  transcription_result
        # return SummarizeResponse(transcript=transcription_result.get("transcript_text", ""), mp3_path=mp3_path)
        return TranscriptResponse(transcript=transcription_result, mp3_path=audio_url)

    
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    
