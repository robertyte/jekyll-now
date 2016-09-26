---
layout: post
title: Howdy, world!
tag: [start, beginning]
---

This is my first post and I am so excited about it.

Woohoo!

<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>

<h2>{{page.title}}</h2>
{% capture tags %}
  {% for tag in site.tags %}
    {{ tag[0] }}
  {% endfor %}
{% endcapture %}
{% assign sortedtags = tags | split:' ' | sort %}

{% for tag in sortedtags %}
  <h3 id="{{ tag | escape }}">{{ tag }}</h3>
  <ul>
  {% for post in site.tags[tag] %}
    <li><a href="{{site.baseurl}}{{ post.url }}">{{ post.title }}</a></li>
  {% endfor %}
  </ul>
{% endfor %}
