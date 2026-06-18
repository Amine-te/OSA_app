from django import forms
from .models import CameraConfig, AlertRule

class CameraConfigForm(forms.ModelForm):
    class Meta:
        model = CameraConfig
        fields = ['name', 'rtsp_url', 'frame_skip', 'confidence_threshold', 'void_confidence_threshold']
        widgets = {
            'confidence_threshold': forms.NumberInput(attrs={'step': '0.05'}),
            'void_confidence_threshold': forms.NumberInput(attrs={'step': '0.05'}),
        }

class AlertRuleForm(forms.ModelForm):
    class Meta:
        model = AlertRule
        fields = '__all__'
