
from http.server import *
import visualization

hostName = ""
serverPort = 80

template = '''
<html>
<meta name="viewport" content="width=device-width, initial-scale=1">
<head>
<style>
.button {
  background-color: #00bfff;
  border: none;
  color: white;
  text-align: center;
  text-decoration: none;
  display: inline-block;
  font-size: 20px;
  margin: 4px 2px;
  cursor: pointer;
  width: 160px; 
  height: 60px;
}
</style>
<title>
Light Strip Control
</title>
</head>
<body>
<center>
:D %s
<form action="" method="GET">
<button class="button" type="submit" name="type" value=1>Energy</button>
<br><br>
<button class="button" type="submit" name="type" value=2>Scroll</button>
<br><br>
<button class="button" type="submit" name="type" value=3>Spectrum</button>
<br><br>
<button class="button" type="submit" name="type" value=0>Off</button>
<br><br>
</form>
</center>
</body>
</html>
'''

class LightStripServer(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/?type=1":
            visualization.start_energy()
            pass
        if self.path == "/?type=0":
            visualization.stop_everything()
            pass
        if self.path == "/?type=2":
            visualization.start_scroll()
            pass
        if self.path == "/?type=3":
            visualization.start_spectrum()
            pass
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        response = template % ":) Stas"
        self.wfile.write(bytes(response, "utf-8"))
        return


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('stas')

    webServer = ThreadingHTTPServer((hostName, serverPort), LightStripServer)
    webServer.serve_forever()
    webServer.server_close()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


