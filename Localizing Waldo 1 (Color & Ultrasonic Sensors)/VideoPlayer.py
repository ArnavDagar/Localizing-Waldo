from moviepy.editor import VideoFileClip
from IPython.display import HTML, display

white_output = r'C:\Users\Arnav\OneDrive\Documents\Intermediate Python 2\TIC TAC TOE FOLDER\GSDSEF\Latest\Project Intro Video Arnav Dagar Junior Divison.mp4'

a = HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format()
