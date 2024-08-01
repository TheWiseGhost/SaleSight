import os
import subprocess
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = 'Runs the Django development server and builds the frontend'

    def handle(self, *args, **kwargs):
        # Get the absolute path to the project root and frontend directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        manage_py_path = os.path.join(project_root, 'manage.py')
        frontend_path = os.path.join(project_root, 'frontend')
         # Run Django development server
        self.stdout.write(self.style.SUCCESS('Starting Django development server...'))
        subprocess.run(['python', manage_py_path, 'runserver'], cwd=project_root, check=True)
        # Run npm build
        self.stdout.write(self.style.SUCCESS('Running npm build...'))
        subprocess.run(['npm', 'run', 'build'], cwd=frontend_path, check=True)
        
       