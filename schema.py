from enum import Enum

class BlogSize(str, Enum):
    Small = "Small"
    Medium = "Medium"
    Large = "Large"
    
    
class BlogTone(str, Enum):
    Engaging = "Engaging"
    Inspirational = "Inspirational"
    Informative = "Informative"
    Professional = "Professional"
    Conversational = "Conversational"
    Promotional = "Promotional"
    Storytelling = "Storytelling"
    Educational = "Educational"
    News = "News"
    Humorous = "Humorous"
    Casual = "Casual"
    Review = "Review"
    HowTo = "How To"
    
class MediaLanguage(str, Enum):
    English = "English"
    Spanish = "Spanish"
    French = "French"
    German = "German"
    Italian = "Italian"
    Portuguese = "Portuguese"
    Dutch = "Dutch"
    Russian = "Russian"
    Chinese = "Chinese"
    Japanese = "Japanese"
    Korean = "Korean"
    Arabic = "Arabic"
    Hindi = "Hindi"
    Bengali = "Bengali"
    Turkish = "Turkish"
    Polish = "Polish"
    Swedish = "Swedish"
    Danish = "Danish"
    Norwegian = "Norwegian"
    Finnish = "Finnish"
        # Add more languages here
        
class BlogLanguage(str, Enum):
    English = "English"
    Spanish = "Spanish"
    French = "French"
    German = "German"
    Italian = "Italian"
    Portuguese = "Portuguese"
    Dutch = "Dutch"
    Russian = "Russian"
    Chinese = "Chinese"
    Japanese = "Japanese"
    Korean = "Korean"
    Arabic = "Arabic"
    Hindi = "Hindi"
    Bengali = "Bengali"
    Turkish = "Turkish"
    Polish = "Polish"
    Swedish = "Swedish"
    Danish = "Danish"
    Norwegian = "Norwegian"
    Finnish = "Finnish"
    # Add more languages here
    
class BlogGenerationMode(str, Enum):
    AutoPilot = "auto-Pilot"
class WriterPointOfView(str, Enum):
    FirstPerson = "First Person (I and We)"
    SecondPerson = "Second Person (You)"
    ThirdPerson = "Third Person (He, She, They, It)"
    
class BlogCoverImg(str, Enum):
    URL = "url"
