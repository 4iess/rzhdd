import pyttsx3
import asyncio



async def object_voice(object_name):
    tts = pyttsx3.init()

    voices = tts.getProperty('voices')

    # Задать голос по умолчанию
    tts.setProperty('voice', 'ru')

    # Попробовать установить предпочтительный голос
    for voice in voices:
        if voice.name == 'Aleksandr':
            tts.setProperty('voice', voice.id)

    await asyncio.to_thread(tts.say, object_name)  # Асинхронно вызываем tts.say
    await asyncio.to_thread(tts.runAndWait)  # Асинхронно вызываем tts.runAndWait