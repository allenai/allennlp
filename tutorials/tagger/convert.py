import re

FILENAME = 'tutorials/tagger/basic_allennlp_annotated.py'

full_text = open(FILENAME).read()

comment_regex = r"[ ]*(####.*)\n"

parts = re.split(comment_regex, full_text)

HTML = parts[0] + '\n<table class="annotated-code">'

parts = parts[1:]
num_parts = len(parts) // 2

for i in range(num_parts):
    comment = parts[2 * i]
    comment = comment.replace("####", "").strip()
    code = parts[2 * i + 1]
    code = code.rstrip("\n")

    row = f"""
    <tr>
        <td class="code">
{{% highlight python %}}
{code}
{{% endhighlight %}}
        </td>
        <td class="desc">
            {comment}
        </td>
    </tr>
    """

    HTML += row

HTML += "</table>"

with open('tutorials/tagger/table.html', 'w') as f:
    f.write(HTML)
