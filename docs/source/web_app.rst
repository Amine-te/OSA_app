=============================================
OSA-Web: Fleet Monitoring Dashboard
=============================================

Overview
========

The **OSA-Web** dashboard is a comprehensive Django-based application designed for multi-camera fleet monitoring. It utilizes real-time WebSocket frame streaming and a Celery task queue to perform AI-powered shelf analysis asynchronously, allowing centralized management of entire retail spaces.

Features
========

* **Multi-Camera Management**: Easily configure and monitor numerous RTSP streams simultaneously.
* **Real-time WebSockets**: Streams annotated frames and live analytics directly to the browser with ultra-low latency.
* **Distributed Processing**: Utilizes Redis and Celery to execute heavy computer-vision pipelines asynchronously.
* **AI Assistant Co-Pilot**: Integrated intelligent assistant capable of understanding natural language queries and generating dynamic charts based on inventory data.

User Interface
==============

Home Dashboard
--------------

The Web Dashboard Home provides a centralized view of all active monitoring sessions, fleet-wide alerts, and top-level KPIs.

.. figure:: _static/screenshots/osa-web-screenshots/home.jpg
   :alt: OSA-Web Home Dashboard
   :width: 100%
   :align: center
   :class: presentation-image

Real-Time Monitoring
--------------------

Watch a live, annotated feed of any active camera session with associated real-time metrics streaming alongside the video.

.. figure:: _static/screenshots/osa-web-screenshots/real_time_monitoring.jpg
   :alt: Real-Time Monitoring Feed
   :width: 100%
   :align: center
   :class: presentation-image

Configuration
-------------

Administrators can easily configure camera endpoints, model paths, and alert rules through the management interface.

.. figure:: _static/screenshots/osa-web-screenshots/configuration.jpg
   :alt: Camera and Alert Configuration
   :width: 100%
   :align: center
   :class: presentation-image

Session Management
==================

Sessions History
----------------

Browse past monitoring sessions, review performance data, and access historical analytics.

.. figure:: _static/screenshots/osa-web-screenshots/sessions_history.jpg
   :alt: Sessions History
   :width: 100%
   :align: center
   :class: presentation-image

Session Details
---------------

Examine a specific session in depth, exploring the timeline of events, threshold triggers, and overall performance during that period.

.. figure:: _static/screenshots/osa-web-screenshots/session_details.jpg
   :alt: Session Details
   :width: 100%
   :align: center
   :class: presentation-image

Analytics & Reporting
===================

KPIs Dashboard
--------------

Track aggregated Key Performance Indicators across the entire store or drill down into specific camera zones.

.. figure:: _static/screenshots/osa-web-screenshots/kpis.jpg
   :alt: Key Performance Indicators
   :width: 100%
   :align: center
   :class: presentation-image

Stock Inventory
---------------

View current stock levels, missing items, and capacity data for all monitored product categories.

.. figure:: _static/screenshots/osa-web-screenshots/stock.jpg
   :alt: Stock Inventory Table
   :width: 100%
   :align: center
   :class: presentation-image

Temporal Analysis
-----------------

Analyze stock fluctuations and restock cycles over time using interactive temporal charts.

.. figure:: _static/screenshots/osa-web-screenshots/temporal_analysis.jpg
   :alt: Temporal Analysis Charts
   :width: 100%
   :align: center
   :class: presentation-image

AI Assistant Co-Pilot
=====================

The OSA-Web dashboard includes a powerful Groq-powered AI Assistant. 

Interactive Queries
-------------------

Ask natural language questions about your inventory data (e.g., "Which products are critically low?", "Show me the restocking trends for Cola").

.. figure:: _static/screenshots/osa-web-screenshots/ai_assistant.jpg
   :alt: AI Assistant Chat Interface
   :width: 100%
   :align: center
   :class: presentation-image

Dynamic Graph Generation
------------------------

The AI Assistant can generate dynamic, interactive graphs directly in the chat interface in response to complex analytical questions.

.. figure:: _static/screenshots/osa-web-screenshots/ai_assistant_graph_response.jpg
   :alt: AI Assistant Graph Response
   :width: 100%
   :align: center
   :class: presentation-image
