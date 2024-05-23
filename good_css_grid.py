from IPython.display import display, HTML
import pandas as pd
import lorem

# Generate a list of lorem ipsum sentences
lorem_texts = [lorem.sentence() for _ in range(10)]  # Generate 10 lorem ipsum sentences

# Create a DataFrame
df = pd.DataFrame(lorem_texts, columns=["Lorem Ipsum"])

html_table = df[["Lorem Ipsum"]].to_html(index=False, border=0)


html = """
<style>
.grid-container {
  display: grid;
  grid-template-columns: auto auto auto; /* Three columns of equal width */
  grid-template-rows: auto auto auto auto auto; /* Five rows */
  gap: 10px; /* Space between rows and columns */
  background-color: #f4f4f4; /* Light grey background */
  padding: 10px; /* Padding around the grid */
}

.grid-item {
  background-color: #ffffff;
  border: 1px solid rgba(0, 0, 0, 0.8);
  padding: 20px;
  font-size: 16px;
  text-align: center;
  border-radius: 5px; /* Rounded corners */
  transition: transform 0.2s; /* Smooth transform on hover */
}

.grid-item:hover {
  transform: scale(1.05); /* Slightly increase size on hover */
  cursor: pointer;
}

.item1 { grid-row: 1 / 3; background-color: #ffcccb; } /* Light red */
.item2 { background-color: #add8e6; } /* Light blue */
.item3 { background-color: #90ee90; } /* Light green */
.item4 { background-color: #ffb6c1; } /* Light pink */
.item5 { background-color: #ffffe0; } /* Light yellow */
.item6 { background-color: #dda0dd; } /* Plum */
.item7 { background-color: #9acd32; } /* Yellow Green */
.item8 { background-color: #20b2aa; } /* Light Sea Green */
.item9 { background-color: #87cefa; } /* Sky Blue */
.item10 { background-color: #778899; } /* Light Slate Gray */
.item11 { background-color: #f08080; } /* Light Coral */
.item12 { background-color: #66cdaa; } /* Medium Aqua Marine */
.item13 { background-color: #6495ed; } /* Cornflower Blue */
.item14 { background-color: #ff6347; } /* Tomato */
</style>

<div class="grid-container">
  <div class="grid-item item1">Start with one of the following suggested tasks, or ask me anything using the text box below</div>
  <div class="grid-item item2">Rust is a multi-paradigm, general-purpose programming language that emphasizes performance, type safety, and concurrency. It enforces memory safety—meaning that all references point to valid memory—without a garbage </div>
  <div class="grid-item item3">Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured, object-oriented and functional programming.</div>
  <div class="grid-item item4">4</div>
  <div class="grid-item item5">5
  </div>
  <div class="grid-item item6">6</div>
  <div class="grid-item item7">7</div>
  <div class="grid-item item8">8</div>
  <div class="grid-item item9">9</div>
  <div class="grid-item item10">10</div>
  <div class="grid-item item11">11</div>
  <div class="grid-item item12">12</div>
  <div class="grid-item item13">13</div>
  <div class="grid-item item14">14</div>
</div>
"""

display(HTML(html))
