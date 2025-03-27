class fsp():
    def __init__(self,instance):
        self.instname = instance
        self.nbJob,self.nbMach = self.read_nbJob_nbMachine(self.instname)
        self.eval = self.make_eval(self.instname)

    def read_nbJob_nbMachine(self,instname):
        if instname[0]=='t':
            filename=str("./taillard/"+instname+".dat")
        if instname[0]=='V':
            filename=str("./vrf/"+instname+"_Gap.txt")

        with open(filename, 'r') as f:
            content = f.read()
            lines = content.split('\n')

            nbJob = int(lines[0].split()[0])
            nbMach = int(lines[0].split()[1])
            return nbJob,nbMach

    # read from input file
    def read_input(self,instname):
        if instname[0]=='t':
            filename=str("./taillard/"+instname+".dat")
        if instname[0]=='V':
            filename=str("./vrf/"+instname+"_Gap.txt")

        # filename=str(pathlib.Path(__file__).parent.absolute())+"/problemData/fsp/"+instname+".dat"
        with open(filename, 'r') as f:
            content = f.read()
            lines = content.split('\n')

            nbJob = int(lines[0].split()[0])
            nbMach = int(lines[0].split()[1])

            PTM = [[int(0)]*nbJob for i in range(nbMach)]

            if instname[0]=='t':
                #row stores per-machine time
                for i,l in enumerate(lines[1:]):
                    if not l.strip():
                        continue
                    if l.startswith('EOF'):
                        break
                    else:
                        PTM[i] = [int(x) for x in l.split()]

            if instname[0]=='V':
                #row stores per-job time
                for j,l in enumerate(lines[1:]):
                    if not l.strip():
                        continue
                    if l.startswith('EOF'):
                        break
                    else:
                        for i in range(0,nbMach):
                            PTM[i][j]=int(l.split()[2*i+1])

        return PTM,nbJob,nbMach


    def make_eval(self,instance):
        PTM,nJob,nMach = self.read_input(instance)

        def evaluate(perm) -> int:
            assert len(perm) == nJob

            tmp = [0]*nMach
            for i in range(nJob):
                jb = perm[i]
                tmp[0] += PTM[0][jb]
                for j in range(1,nMach):
                    tmp[j] = max(tmp[j],tmp[j-1]) + PTM[j][jb]
            return tmp[nMach-1]
        return evaluate
