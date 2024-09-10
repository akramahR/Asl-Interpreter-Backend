from rest_framework import serializers

class VideoUploadSerializer(serializers.Serializer):
    video = serializers.FileField()

    def validate_video(self, value):
        # You can add custom validation here (e.g., file type, size, etc.)
        return value
