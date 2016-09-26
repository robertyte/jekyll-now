---
layout: post
title: Howdy, world!
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
