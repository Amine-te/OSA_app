import json

from channels.generic.websocket import AsyncWebsocketConsumer


class StreamConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for live session streaming.

    Joins a channel group ``session_<id>`` on connect and forwards
    ``stream.update`` messages (pushed by the Celery task) to the
    browser as JSON.
    """

    async def connect(self):
        self.session_id = self.scope['url_route']['kwargs']['session_id']
        self.group_name = f"session_{self.session_id}"
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def receive(self, text_data=None, bytes_data=None):
        pass  # client never sends

    async def stream_update(self, event):
        """Handle stream.update messages from the channel layer."""
        await self.send(text_data=json.dumps(event['data']))
