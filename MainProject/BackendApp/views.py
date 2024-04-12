from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from django.http import HttpResponse
import os
# Create your views here.
@csrf_exempt
def upload_image(request):
    if request.method == 'POST':
        image_file = request.FILES.get('image')
        print(image_file)
        if image_file:
            # Process the image or save it
            save_path = 'files'
            
            os.makedirs(save_path, exist_ok=True)
            
            with open(os.path.join(save_path, image_file.name), 'wb') as f:
                for chunk in image_file.chunks():
                    f.write(chunk)

            
            return JsonResponse(data={'status': 'success', 'message': 'Image received'})
        else:
            return JsonResponse({'status': 'error', 'message': 'No image received'})

    return JsonResponse({'status': 'error', 'message': 'Invalid request'})
def test(request):
    return HttpResponse("Any kind of HTML Here")