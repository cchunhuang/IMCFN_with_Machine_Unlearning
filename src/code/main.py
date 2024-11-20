import sys

from UnlearnableIMCFN import UnlearnableIMCFN

def main():
    '''
    Main function of UnlearnableIMCFN.
    '''
    if len(sys.argv) >= 2:
        config_path = sys.argv[1]
    else:
        config_path = None
    
    sys.argv = [sys.argv[0]]
    unlearnable_imcfn = UnlearnableIMCFN(config_path)
    
    if unlearnable_imcfn.config.train == True:
        unlearnable_imcfn.trainModel()
    elif unlearnable_imcfn.config.predict == True:
        unlearnable_imcfn.predict()
    elif unlearnable_imcfn.config.unlearn == True:
        unlearnable_imcfn.unlearn()
        
if __name__ == '__main__':
    main()