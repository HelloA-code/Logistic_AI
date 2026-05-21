import { useState, useCallback, useEffect } from 'react';
import { supabase } from '@/integrations/supabase/client';
import { SupportTicket, SupportTicketStatus, SupportTicketType } from '@/lib';

/**
 * Real-time support ticket management with Supabase integration
 */

/**
 * useSupportTickets hook manages the lifecycle of emergency and support requests.
 * It integrates with the browser Geolocation API to capture GPS coordinates for logistics assistance.
 */
export function useSupportTickets() {
  const [tickets, setTickets] = useState<SupportTicket[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch tickets from Supabase
  const fetchTickets = useCallback(async () => {
    setIsLoading(true);
    try {
      const { data, error: fetchError } = await supabase
        .from('support_tickets')
        .select(`
          id,
          ticket_number,
          driver_id,
          ticket_type,
          severity,
          title,
          description,
          location_lat,
          location_lng,
          location_address,
          status,
          created_at
        `)
        .order('created_at', { ascending: false });

      if (fetchError) throw fetchError;

      // Transform data to match interface
      const transformedTickets: SupportTicket[] = (data || []).map(t => ({
        id: t.ticket_number || t.id,
        user_id: t.driver_id,
        type: t.ticket_type?.toLowerCase().replace('_', '-') as SupportTicketType,
        status: t.status as SupportTicketStatus,
        description: t.description,
        location: {
          lat: t.location_lat || 0,
          lng: t.location_lng || 0,
          address: t.location_address || 'Unknown location'
        },
        created_at: t.created_at,
        priority: t.severity as SupportTicket['priority']
      }));

      setTickets(transformedTickets);
    } catch (err) {
      console.error('Error fetching tickets:', err);
      setError('Failed to load support tickets');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Initialize data and set up real-time subscription
  useEffect(() => {
    fetchTickets();

    // Set up real-time subscription
    const subscription = supabase
      .channel('support_tickets_changes')
      .on('postgres_changes',
        { event: '*', schema: 'public', table: 'support_tickets' },
        () => fetchTickets()
      )
      .subscribe();

    return () => {
      subscription.unsubscribe();
    };
  }, [fetchTickets]);

  /**
   * Captures the current GPS coordinates of the device.
   * Falls back to a default location if permission is denied.
   */
  const getCurrentLocation = useCallback(async (): Promise<{ lat: number; lng: number; address?: string }> => {
    return new Promise((resolve) => {
      if (!navigator.geolocation) {
        resolve({ lat: 28.6139, lng: 77.2090, address: 'New Delhi (Fallback)' });
        return;
      }

      navigator.geolocation.getCurrentPosition(
        (position) => {
          resolve({
            lat: position.coords.latitude,
            lng: position.coords.longitude,
            address: 'GPS Verified Location'
          });
        },
        () => {
          // Standard fallback to major hub if GPS access is restricted
          resolve({ lat: 28.6139, lng: 77.2090, address: 'GPS Access Restricted' });
        },
        { timeout: 8000, enableHighAccuracy: true }
      );
    });
  }, []);

  /**
   * Creates a new support ticket with automatic GPS location attachment.
   */
  const createTicket = useCallback(async (data: {
    type: SupportTicketType;
    description: string;
    priority: SupportTicket['priority'];
    title?: string;
  }) => {
    setIsLoading(true);
    setError(null);

    try {
      const locationData = await getCurrentLocation();
      
      // Get current user session
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) throw new Error('User not authenticated');

      // Get driver ID for the current user
      const { data: driverData } = await supabase
        .from('drivers')
        .select('id')
        .eq('user_id', user.id)
        .single();

      const driverId = driverData?.id || '55555555-5555-5555-5555-555555555555'; // Fallback

      // Create ticket via edge function
      const response = await supabase.functions.invoke('logistics_operations_2026_01_31_17_50', {
        body: {
          action: 'create_support_ticket',
          data: {
            driverId,
            ticketType: data.type.toUpperCase().replace('-', '_'),
            severity: data.priority,
            title: data.title || `${data.type.replace('-', ' ')} - ${new Date().toLocaleDateString()}`,
            description: data.description,
            locationLat: locationData.lat,
            locationLng: locationData.lng,
            locationAddress: locationData.address
          }
        }
      });

      if (response.error) throw response.error;

      // Refresh tickets to show the new one
      await fetchTickets();
      
      return response.data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'An error occurred while creating the support ticket';
      setError(message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [getCurrentLocation, fetchTickets]);

  /**
   * Updates the status of an existing ticket (e.g., from Open to In-Progress).
   */
  const updateTicketStatus = useCallback(async (ticketId: string, status: SupportTicketStatus) => {
    setIsLoading(true);
    try {
      const { error: updateError } = await supabase
        .from('support_tickets')
        .update({ 
          status,
          updated_at: new Date().toISOString(),
          ...(status === 'resolved' && { resolved_at: new Date().toISOString() })
        })
        .eq('ticket_number', ticketId);

      if (updateError) throw updateError;
      
      // Update local state immediately
      setTickets((prev) =>
        prev.map((t) => (t.id === ticketId ? { ...t, status } : t))
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update ticket status');
    } finally {
      setIsLoading(false);
    }
  }, []);

  return {
    tickets,
    isLoading,
    error,
    createTicket,
    updateTicketStatus,
    getCurrentLocation,
    fetchTickets
  };
}
