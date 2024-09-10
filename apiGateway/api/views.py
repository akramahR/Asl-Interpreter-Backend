from django.shortcuts import render

# Create your views here.
from .src.inference import test_model_on_video
from .apps import *
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import VideoUploadSerializer
from .apps import ApiConfig
from sklearn.preprocessing import LabelEncoder
import tempfile


def upload_page(request):
    return render(request, 'upload.html')


class Prediction(APIView):
    def post(self, request):
        serializer = VideoUploadSerializer(data=request.data)
        if serializer.is_valid():
            video_file = serializer.validated_data['video']

            # Save the uploaded video to a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_video:
                for chunk in video_file.chunks():
                    temp_video.write(chunk)
                temp_video_path = temp_video.name

            # Process the video and pass it to your model
            model = ApiConfig.model
            label_encoder = LabelEncoder()
            label_classes = ApiConfig.label_classes
            label_encoder.classes_ = label_classes

            predicted_gesture = test_model_on_video(model, temp_video_path, label_encoder)

            os.remove(temp_video_path)

            return Response({'result': predicted_gesture}, status=status.HTTP_200_OK)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
