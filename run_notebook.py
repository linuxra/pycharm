import sys
import json
from subprocess import call
from multiprocessing import Process

def convert_notebook(notebook_name, output_html_prefix, param):
    """
    Convert a specific Jupyter notebook to HTML based on a single date parameter.

    Parameters:
    - notebook_name (str): The name/path of the Jupyter notebook to be converted.
    - output_html_prefix (str): The prefix for the output HTML file.
    - param (str): The date parameter, used to tailor the notebook execution and naming the output HTML.

    This function writes the date parameter to a temporary JSON file, then uses nbconvert to
    execute the notebook and produce an HTML output.
    """
    # Write the single parameter to a temporary JSON file
    with open('temp_params_{}.json'.format(param), 'w') as f:
        json.dump({"date": param}, f)

    # Execute the Jupyter notebook and convert to HTML using the config file
    call(["jupyter", "nbconvert", "--to", "html", "--execute", "--config", "nbconvert_config.py", notebook_name, "--output", "{}_{}.html".format(output_html_prefix, param)])

    # Optional: remove the temp_params_{}.json file after execution
    # call(["rm", "temp_params_{}.json".format(param)])


def main():
    """
    Main execution function for the script.

    Usage:
    python run_notebook.py [notebook_name] [output_html_prefix] [param1] [param2] ...

    Where:
    - notebook_name: The name/path of the Jupyter notebook to be converted.
    - output_html_prefix: The prefix for the output HTML files.
    - paramX: The list of date parameters. For each date, a separate process is spawned to execute the notebook
              and produce an HTML file named as [output_html_prefix]_[param].html

    This function creates separate processes for each date parameter and initiates the conversion in parallel.
    """
    if len(sys.argv) < 4:
        print("Usage: python run_notebook.py [notebook_name] [output_html_prefix] [param1] [param2] ...")
        return

    notebook_name = sys.argv[1]
    output_html_prefix = sys.argv[2]
    params = sys.argv[3:]

    # Create processes for each parameter and start them
    processes = []
    for param in params:
        p = Process(target=convert_notebook, args=(notebook_name, output_html_prefix, param))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
