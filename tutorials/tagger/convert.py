import re

FILENAME = 'tutorials/tagger/basic_allennlp.py'

full_text = open(FILENAME).read()

comment_regex = r"[ ]*(####.*)\n"

parts = re.split(comment_regex, full_text)

# remove newlines, then triple quotes, then more newlines
HTML = parts[0].strip().strip('"""').strip()

parts = parts[1:]
num_parts = len(parts) // 2

comments = parts[::2]
codes = parts[1::2]

HTML += """<div id="annotated-code">
  <!-- Code Blocks -->
  <div class="annotated-code__pane annotated-code__pane--code-container">
"""

for i, code in enumerate(codes):
    code = code.rstrip("\n")
    HTML += f"""<div class="annotated-code__code-block" id="c{i}">
{{% highlight python %}}
{code}
{{% endhighlight %}}
</div>
"""

HTML += """</div>
    <!-- END Code Blocks -->

    <!-- Annotations -->
    <div class="annotated-code__pane annotated-code__pane--annotations-container">
        <ul id="annotated-code__annotations">
"""

for i, comment in enumerate(comments):
    comment = comment.strip("####").strip()
    HTML += f"""<li class="annotation" id="a{i}">{comment}</li>
"""

HTML += """</ul>
  </div><!-- END Annotations -->
</div><!-- END Annotated Code -->
 {% include more-tutorials.html %}
"""

with open('tutorials/tagger/default-tutorial.html', 'w') as f:
    f.write(HTML)
