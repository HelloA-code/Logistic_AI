import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Session, AuthChangeEvent, User } from '@supabase/supabase-js';
import { supabase } from '@/integrations/supabase/client';
import { ROUTE_PATHS, UserProfile, UserRole } from '@/lib/index';

/**
 * Authentication hook for role-based login system with Supabase integration.
 * Manages the lifecycle of user sessions for Operators, Drivers, and Suppliers
 * within the AI Path Logistics ecosystem.
 */
export function useAuth() {
  const [user, setUser] = useState<UserProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  /**
   * Builds a UserProfile from a Supabase auth user object (metadata-based)
   */
  const buildProfileFromAuthUser = useCallback((authUser: User): UserProfile => {
    const meta = authUser.user_metadata || {};
    // Use metadata role, then localStorage fallback, then default
    const savedRole = localStorage.getItem('ai_path_user_role') as UserRole | null;
    const role = (meta.role as UserRole) || savedRole || 'operator';
    return {
      id: authUser.id,
      email: authUser.email || '',
      full_name: meta.full_name || 'Logistics Professional',
      role,
      company_name: meta.company_name || 'Indian Logistics Network',
      phone: meta.phone || '',
    };
  }, []);

  useEffect(() => {
    let isMounted = true;

    // Initial session check on mount
    const checkSession = async () => {
      try {
        const { data: { session } } = await supabase.auth.getSession();
        if (session?.user && isMounted) {
          setUser(buildProfileFromAuthUser(session.user));
        }
      } catch (error) {
        console.error('Session check failed:', error);
      } finally {
        if (isMounted) setLoading(false);
      }
    };

    checkSession();

    // Subscribe to authentication state changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (event: AuthChangeEvent, session: Session | null) => {
        if (!isMounted) return;

        if (event === 'SIGNED_IN' && session?.user) {
          setUser(buildProfileFromAuthUser(session.user));
          setLoading(false);
        } else if (event === 'SIGNED_OUT') {
          setUser(null);
          setLoading(false);
        } else if (event === 'USER_UPDATED' && session?.user) {
          setUser(buildProfileFromAuthUser(session.user));
          setLoading(false);
        }
      }
    );

    return () => {
      isMounted = false;
      subscription.unsubscribe();
    };
  }, [buildProfileFromAuthUser]);

  /**
   * Executes role-based login logic
   */
  const login = async (email: string, password: string, role: UserRole) => {
    setLoading(true);
    try {
      // Try standard Supabase sign-in first
      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password,
      });

      if (error) {
        // If user doesn't exist, try creating the demo account
        console.warn('Sign-in failed, attempting demo user creation:', error.message);
        
        const { data: signUpData, error: signUpError } = await supabase.auth.signUp({
          email,
          password,
          options: {
            data: {
              full_name: role === 'operator' ? 'Fleet Operator' : role === 'driver' ? 'Fleet Driver' : 'Load Supplier',
              role,
              company_name: 'AI Path Logistics',
              phone: '+91-9876543210',
            },
          },
        });

        if (signUpError) {
          // If both sign-in and sign-up fail, create a local-only demo session
          console.warn('Supabase auth unavailable, using local demo session');
          const demoUser: UserProfile = {
            id: `demo-${role}-${Date.now()}`,
            email,
            full_name: role === 'operator' ? 'Demo Operator' : role === 'driver' ? 'Demo Driver' : 'Demo Supplier',
            role,
            company_name: 'AI Path Logistics (Demo)',
            phone: '+91-9876543210',
          };
          setUser(demoUser);
          localStorage.setItem('ai_path_user_role', role);
          navigate(ROUTE_PATHS.DASHBOARD);
          return;
        }

        if (signUpData.user) {
          setUser(buildProfileFromAuthUser(signUpData.user));
          navigate(ROUTE_PATHS.DASHBOARD);
          return;
        }
      }

      if (data?.user) {
        localStorage.setItem('ai_path_user_role', role);
        setUser(buildProfileFromAuthUser(data.user));
        navigate(ROUTE_PATHS.DASHBOARD);
      }
    } catch (error) {
      console.error('Authentication attempt failed:', error);
      throw error;
    } finally {
      setLoading(false);
    }
  };

  /**
   * Terminates the current session and redirects to landing page
   */
  const logout = async () => {
    setLoading(true);
    try {
      await supabase.auth.signOut();
      setUser(null);
      localStorage.removeItem('ai_path_user_role');
      navigate(ROUTE_PATHS.HOME);
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      setLoading(false);
    }
  };

  return {
    user,
    loading,
    login,
    logout,
    isAuthenticated: !!user,
    isOperator: user?.role === 'operator',
    isDriver: user?.role === 'driver',
    isSupplier: user?.role === 'supplier',
    supabase
  };
}
