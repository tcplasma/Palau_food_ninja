import pyttsx3
import threading
import queue
import time
import os
import hashlib
import numpy as np
import pygame
from foodninja.core.utils import get_resource_path

class VoiceAnnouncer:
    """
    Threaded voice announcer that caches speech to disk to reduce CPU load.
    """
    def __init__(self, cache_dir: str = "assets/audio"):
        self.queue = queue.Queue()
        self.running = True
        self.cache_dir = get_resource_path(cache_dir)
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        self.cached_sounds = {}
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def announce(self, text: str):
        """Add a message to the announcement queue, clearing pending ones."""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self.queue.put(text)

    def _run(self):
        try:
            # Main engine for synthesis (only used when cache miss)
            engine = pyttsx3.init()
            engine.setProperty('rate', 165) 
            engine.setProperty('volume', 1.0)
            voices = engine.getProperty('voices')
            for voice in voices:
                if "EN-US" in voice.id.upper() or "ENGLISH" in voice.name.upper():
                    engine.setProperty('voice', voice.id)
                    break
        except Exception as e:
            print(f"Audio Engine Error: {e}")
            return

        while self.running:
            try:
                text = self.queue.get(timeout=0.1)
                
                # Generate a unique filename for this text
                text_hash = hashlib.md5(text.encode()).hexdigest()
                file_path = os.path.join(self.cache_dir, f"{text_hash}.wav")
                
                # Synthesize if not in cache folder
                if not os.path.exists(file_path):
                    engine.save_to_file(text, file_path)
                    engine.runAndWait()
                
                # Play using Pygame (much lower CPU than pyttsx3 runAndWait)
                if file_path not in self.cached_sounds:
                    try:
                        self.cached_sounds[file_path] = pygame.mixer.Sound(file_path)
                    except Exception as e:
                        print(f"Error loading cached sound: {e}")
                        continue
                
                self.cached_sounds[file_path].play()
                time.sleep(0.01) # Small breathe for the audio mixer
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Announcement Error: {e}")

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)


class SfxEngine:
    """
    Generates and plays synthetic sound effects. Fixes the stereo mixer requirement.
    """
    def __init__(self):
        if not pygame.mixer.get_init():
            pygame.mixer.pre_init(44100, -16, 2, 512)
        
        # Increase simultaneous channels to avoid skipping sounds
        pygame.mixer.set_num_channels(16)
            
        self.correct_sound = self._generate_slice_sound(880, 220, 0.15)
        self.wrong_sound = self._generate_slice_sound(440, 110, 0.25)
        self.bomb_sound = self._generate_explosion_sound(0.4)

    def _generate_slice_sound(self, start_freq, end_freq, duration, sample_rate=44100):
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        freqs = np.geomspace(start_freq, end_freq, len(t))
        phases = 2 * np.pi * np.cumsum(freqs) / sample_rate
        wave = 0.5 * np.sin(phases)
        fade_out = np.linspace(1.0, 0, len(t))
        wave *= fade_out
        
        # Scale to 16-bit
        audio_data = (wave * 32767).astype(np.int16)
        
        # Fix: Convert to 2D for stereo mixer if necessary
        # Most modern systems initialize Pygame mixer in stereo.
        stereo_data = np.repeat(audio_data[:, np.newaxis], 2, axis=1)
        
        return pygame.sndarray.make_sound(stereo_data)

    def _generate_explosion_sound(self, duration, sample_rate=44100):
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        # White noise
        noise = np.random.uniform(-1, 1, len(t))
        # Low pass filter effect via frequency roll-off
        fade_out = np.exp(-5 * t / duration)
        wave = noise * fade_out
        audio_data = (wave * 32767).astype(np.int16)
        stereo_data = np.repeat(audio_data[:, np.newaxis], 2, axis=1)
        return pygame.sndarray.make_sound(stereo_data)

    def play_correct(self):
        self.correct_sound.play()

    def play_wrong(self):
        self.wrong_sound.play()

    def play_bomb(self):
        self.bomb_sound.stop() # Ensure it restarts from the beginning
        self.bomb_sound.play()
