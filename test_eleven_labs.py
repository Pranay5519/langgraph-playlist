import os
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import save

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("ELEVEN_API_KEY")

# Create client
client = ElevenLabs(api_key=api_key)

# Generate audio
audio = client.text_to_speech.convert(
    voice_id="21m00Tcm4TlvDq8ikWAM",  # Example voice ID (Rachel)
    model_id="eleven_multilingual_v2",
    text="Hello Pranay! This is your AI voice assistant speaking.",
)

# Save to file
save(audio, "output.mp3")

print("Audio saved as output.mp3")