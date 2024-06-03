classdef testMembraneSimulation < matlab.unittest.TestCase
    properties
        u
        m
    end
    
    methods (TestMethodSetup)
        function createObjects(testCase)
            % Create 'unit' and 'membrane' objects
            testCase.u = ComUnit('erg', ComUnit.nm_to_cm(1000), 300, ComUnit.kBT_to_erg(10, 300));
            testCase.m = ModMembrane(2, 'unit', testCase.u);
            testCase.m.pm.Vdh.V0 = 0.1;
            testCase.m.pm.k_c = 1;
        end
    end
    
    methods (Test)
        function testInternalForce(testCase)
            % Test internal force calculation
            Fi = Finternal(testCase.m, 'plot_or_not', false);
            testCase.verifyNotEmpty(Fi, 'Internal force calculation failed.');
        end
        
        function testVolumeCalculation(testCase)
            % Test volume calculation
            vol = sum(Volume(testCase.m));
            testCase.verifyGreaterThan(vol, 0, 'Volume calculation failed.');
        end
        
        function testSurfaceAreaCalculation(testCase)
            % Test surface area calculation
            area = sum(Area(testCase.m));
            testCase.verifyGreaterThan(area, 0, 'Surface area calculation failed.');
        end
        
        function testBendingForceCalculation(testCase)
            % Test bending force calculation
            Fb = zeros(size(testCase.m.var.coord));
            epsilon = 1e-3;
            parfor i = 1:length(testCase.m.var.coord)
                for dim = 1:3
                    m_update = testCase.m;
                    r_orig = m_update.var.coord(i, dim);
                    % Perturb positively
                    m_update.var.coord(i, dim) = r_orig + epsilon;
                    Hp = Helfrich(m_update);
                    % Perturb negatively
                    m_update.var.coord(i, dim) = r_orig - epsilon;
                    Hm = Helfrich(m_update);
                    % Reset the coordinate
                    m_update.var.coord(i, dim) = r_orig;
                    % Finite difference approximation of the gradient
                    Fb(i, dim) = -(sum(Hp) - sum(Hm)) / (2 * epsilon);
                end
            end
            testCase.verifyNotEmpty(Fb, 'Bending force calculation failed.');
        end
    end
end
