from django.shortcuts import render
from django.http import JsonResponse
from baby_name_app.predict import generate

# Create your views here.
def index(request):
    baby_names = generate()
    return JsonResponse({'baby_names': str(baby_names)}, safe=False)
