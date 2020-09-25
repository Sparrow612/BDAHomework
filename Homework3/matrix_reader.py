import base64
import requests
import json
import re


def atom_to_elem(atom: str) -> float:
    if atom.isdigit():
        return float(atom)
    else:
        match = re.match(r'\\frac\{(\d+)}{(\d+)}', atom)
        return int(match.group(1)) / int(match.group(2))


class MatrixReader:
    def __init__(self, img_path):
        self.img_path = img_path
        self.api = "https://www.latexlive.com:5001/api/mathpix/posttomathpix"
        self.latex = self.get_latex()
        self.texList = []
        self.matrix = list()

    def encode_img_base64(self) -> str:
        with open(self.img_path, 'rb') as f:
            img = f.read()
            img_in_base64 = base64.b64encode(img)
            return str(img_in_base64, 'utf-8')

    def get_latex(self) -> str:
        img_in_base64 = self.encode_img_base64()  # base64编码后的图片数据
        header = {
            "src": "data:image/png;base64," + img_in_base64
        }  # request payload
        res = requests.post(url=self.api, json=header)
        return json.loads(res.text).get('latex_styled')

    def latex_to_matrix(self):
        pattern = re.compile(r'\\begin{array}{.*}[\w\W]*\\end{array}', re.DOTALL)
        tex = pattern.findall(self.latex)[0]
        cnt = re.sub(r'(\\begin{array}{cccc}\n|\n\\end{array})', "", tex)
        self.texList = cnt.split('\\\\\n')
        for line in self.texList:
            self.line_to_array(line)
        return self.matrix

    def line_to_array(self, line):
        line = line.replace(' ', '')
        atoms = line.split('&')
        res = list()
        for atom in atoms:
            res.append(atom_to_elem(atom))
        self.matrix.append(res)


if __name__ == '__main__':
    print(MatrixReader('res/matrix.png').latex_to_matrix())
